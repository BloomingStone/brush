#ifndef BRUSH_FIT_H
#define BRUSH_FIT_H
/**
 * brush-fit C ABI (libbrush_fit.so / libbrush_fit.dylib / brush_fit.dll).
 *
 * 训练配置以 JSON 字符串传入 (FitConfig 的 serde 序列化, snake_case 字段,
 * 缺失字段用默认值)。参考字段: dcm, out, mode ("static"|"deform"), points,
 * iters, refine_every, eval, save_ply, save_deform, voxel_mm, ... — 全部
 * 字段见 `brush_fit_example_config` 输出的示例 JSON。
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Windows 动态库导出/导入: Rust cdylib 自动 dllexport 符号, 头文件侧
 * dllimport 供 MSVC 调用方链接 import lib; 非 Windows 为空。 */
#if defined(_WIN32) && defined(BRUSH_FIT_BUILD)
#define BRUSH_FIT_API __declspec(dllexport)
#elif defined(_WIN32)
#define BRUSH_FIT_API __declspec(dllimport)
#else
#define BRUSH_FIT_API
#endif

#define BRUSH_FIT_OK 0
#define BRUSH_FIT_ERR 1

/** 进度回调: iter=0, total=0, loss=NaN 表示训练结束 (Done)。 */
typedef void (*brush_fit_progress_cb)(uint32_t iter, uint32_t total, float loss,
                                      void* user_data);

/**
 * 注册进度回调 (全局, 训练开始前调用; 线程安全)。
 * user_data 为不透明指针, 原样回传回调。
 */
BRUSH_FIT_API void brush_fit_set_progress_cb(brush_fit_progress_cb cb, void* user_data);

/**
 * 训练入口 (同步阻塞直到完成)。
 *
 * @param config_json  NUL 结尾的 FitConfig JSON (见 brush_fit_example_config)。
 * @param errbuf       出错时错误消息写入此缓冲; 可传 NULL。
 * @param errbuf_len   errbuf 容量 (字节)。
 * @return BRUSH_FIT_OK (0) 成功; BRUSH_FIT_ERR (1) 失败 (errbuf 含原因)。
 *
 * 输出 (由 config_json 中 "out" 指定目录):
 *   volume_phase00.nii.gz   (必须, phase=0 volume)
 *   canonical_final.ply     (save_ply=true)
 *   deform_final.bin + deform_field_phase{p:02}.nii.gz (save_deform=true)
 *   canonical_final_{transforms,raw}.bin (save_bin=true)
 */
BRUSH_FIT_API int brush_fit_run(const char* config_json, char* errbuf, size_t errbuf_len);

/**
 * 生成示例 FitConfig JSON (全部字段 + 默认值), 写入 errbuf; 返回 0。
 */
BRUSH_FIT_API int brush_fit_example_config(char* errbuf, size_t errbuf_len);

#ifdef __cplusplus
}
#endif

#endif /* BRUSH_FIT_H */
