//! C FFI 薄层 (cdylib): `brush_fit_run(config_json, out_dir, errbuf, len) -> i32`。
//!
//! - config: [`FitConfig`] 的 JSON (serde, kebab-case 字段)。
//! - 同步阻塞直到训练完成; 返回 0 = 成功, 非 0 = 失败 (errbuf 填错误消息)。
//! - 可选进度回调: `brush_fit_set_progress_cb(cb, user_data)` 注册
//!   `extern "C" fn(iter: u32, total: u32, loss: f32, user_data)`。
//!
//! panic 不允许跨 extern "C" 边界 unwind (会 abort 进程): 用 catch_unwind
//! 捕获转错误码 (同 apps/brush-c 模式)。

use std::ffi::{CStr, c_char, c_void};

use crate::config::FitConfig;
use crate::train::{FitProgress, run_deform_with_progress, run_static_with_progress};

/// 运行结果 (0 = 成功)。
pub const BRUSH_FIT_OK: i32 = 0;
pub const BRUSH_FIT_ERR: i32 = 1;

/// 进度回调签名。
pub type BrushFitProgressCb = extern "C" fn(iter: u32, total: u32, loss: f32, user_data: *mut c_void);

static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
static PROGRESS_CB: std::sync::OnceLock<(BrushFitProgressCb, usize)> = std::sync::OnceLock::new();

/// 注册进度回调 (线程安全, 全局一次; 训练开始前调用)。
#[unsafe(no_mangle)]
pub extern "C" fn brush_fit_set_progress_cb(cb: BrushFitProgressCb, user_data: *mut c_void) {
    let _ = PROGRESS_CB.set((cb, user_data as usize));
}

/// 把错误消息写入 errbuf (截断到 len-1 + NUL)。
unsafe fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    // SAFETY: 调用方保证 buf 指向 len 字节可写内存。
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

fn run_impl(config_json: &str) -> anyhow::Result<crate::train::FitOutcome> {
    let cfg: FitConfig = serde_json::from_str(config_json)
        .map_err(|e| anyhow::anyhow!("config JSON 解析失败: {e}"))?;
    if cfg.dcm.as_os_str().is_empty() {
        anyhow::bail!("config 缺少 dcm 字段");
    }

    // 训练开始前读取已注册回调 (注册发生在别的线程; OnceCell set 后 get 立即可见)。
    let (cb, user_data) = match PROGRESS_CB.get() {
        Some(&(cb, ud)) => (Some(cb), ud as *mut c_void),
        None => (None, std::ptr::null_mut()),
    };

    let mode = cfg.mode;
    let rt = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime")
    });

    let progress = if let Some(cb) = cb {
        let ud = user_data as usize;
        Some(Box::new(move |p: FitProgress| match p {
            FitProgress::Step { iter, total, loss } => {
                cb(iter, total, loss, ud as *mut c_void);
            }
            FitProgress::Done => {
                cb(0, 0, f32::NAN, ud as *mut c_void);
            }
        }) as crate::train::ProgressFn)
    } else {
        None
    };

    match mode {
        crate::config::FitMode::Static => match progress {
            Some(p) => rt.block_on(run_static_with_progress(cfg, p)),
            None => rt.block_on(crate::train::run_static(cfg)),
        },
        crate::config::FitMode::Deform => match progress {
            Some(p) => rt.block_on(run_deform_with_progress(cfg, p)),
            None => rt.block_on(crate::train::run_deform(cfg)),
        },
    }
}

/// 训练入口 (C): `config_json` = FitConfig JSON, `errbuf` 出错时填消息。
/// 返回 BRUSH_FIT_OK (0) 成功 / BRUSH_FIT_ERR (1) 失败。
///
/// # Safety
/// - `config_json` 必须是非空、NUL 结尾的 C 字符串, 调用期间保持有效。
/// - `errbuf` 若非空必须指向至少 `errbuf_len` 字节可写内存。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_fit_run(
    config_json: *const c_char,
    errbuf: *mut c_char,
    errbuf_len: usize,
) -> i32 {
    if config_json.is_null() {
        // SAFETY: errbuf 由调用方按约定保证有效 (允许 null)。
        unsafe { write_err(errbuf, errbuf_len, "config_json is null") };
        return BRUSH_FIT_ERR;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: config_json 已检查非空, 调用方保证是合法 C 字符串。
        let json = unsafe { CStr::from_ptr(config_json) }
            .to_string_lossy()
            .into_owned();
        run_impl(&json)
    }));
    match result {
        Ok(Ok(_)) => BRUSH_FIT_OK,
        Ok(Err(e)) => {
            // SAFETY: errbuf 由调用方按约定保证有效。
            unsafe { write_err(errbuf, errbuf_len, &e.to_string()) };
            BRUSH_FIT_ERR
        }
        Err(_) => {
            // SAFETY: errbuf 由调用方按约定保证有效。
            unsafe { write_err(errbuf, errbuf_len, "internal panic during training") };
            BRUSH_FIT_ERR
        }
    }
}

/// 工具: 把当前配置 schema 的示例 JSON 写入 errbuf (供调用方参考), 返回 0。
///
/// # Safety
/// - `errbuf` 若非空必须指向至少 `errbuf_len` 字节可写内存。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_fit_example_config(
    errbuf: *mut c_char,
    errbuf_len: usize,
) -> i32 {
    let example = serde_json::to_string_pretty(&FitConfig::default())
        .unwrap_or_else(|_| "{}".to_owned());
    // SAFETY: 调用方保证 errbuf 有效。
    unsafe { write_err(errbuf, errbuf_len, &example) };
    BRUSH_FIT_OK
}
