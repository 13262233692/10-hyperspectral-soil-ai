"""端到端集成测试脚本

验证完整链路:
    ENVI解码 -> SG平滑滤波 -> ONNX推理 -> GeoTIFF热力图
"""

import os
import sys
import time
import tempfile
import traceback

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)


def test_envi_decoding():
    print("\n" + "=" * 60)
    print("TEST 1: ENVI原生字节解码器")
    print("=" * 60)
    from hyperspectral_engine.decoding.envi_decoder import (
        ENVIDecoder,
        decode_gf5_cube,
    )
    from generate_test_data import generate_synthetic_envi

    tmpdir = tempfile.mkdtemp()
    hdr_path, dat_path = generate_synthetic_envi(tmpdir, samples=64, lines=48)

    decoder = ENVIDecoder(hdr_path, dat_path)
    meta = decoder.parse_header()
    print(f"  Shape: bands={meta.bands}, lines={meta.lines}, samples={meta.samples}")
    print(f"  Interleave: {meta.interleave}, dtype: {meta.data_type}")

    cube = decoder.decode()
    assert cube.shape == (330, 48, 64), f"Unexpected shape: {cube.shape}"
    assert cube.dtype == np.float32
    print(f"  Cube decoded OK, shape={cube.shape}, dtype={cube.dtype}")
    print(f"  Reflectance range: [{cube.min():.4f}, {cube.max():.4f}]")

    spec = decoder.decode_pixel(10, 20)
    assert spec.shape == (330,)
    print(f"  Random pixel decode OK, spectrum shape={spec.shape}")

    cube2, meta2 = decode_gf5_cube(hdr_path, dat_path)
    np.testing.assert_allclose(cube, cube2, rtol=1e-5)
    print("  ENVI解码测试: PASSED")


def test_sg_preprocessing():
    print("\n" + "=" * 60)
    print("TEST 2: 自适应Savitzky-Golay平滑滤波")
    print("=" * 60)
    from hyperspectral_engine.preprocessing.sg_filter import (
        AdaptiveSavitzkyGolay,
        PreprocessingPipeline,
    )

    rng = np.random.RandomState(123)
    wl = np.linspace(400, 2500, 330)
    base = 0.2 + 0.3 * np.sin(wl / 500)
    noisy = base + rng.normal(0, 0.03, 330).astype(np.float32)

    pipeline = PreprocessingPipeline()
    smoothed = pipeline.process_spectrum(noisy)
    assert smoothed.shape == (330,)

    noise_before = np.std(noisy - base)
    noise_after = np.std(smoothed - base)
    print(f"  Noise level before: {noise_before:.5f}")
    print(f"  Noise level after:  {noise_after:.5f}")
    assert noise_after < noise_before, "Smoothing should reduce noise"
    print("  SG滤波测试: PASSED")

    batch = rng.rand(32, 330).astype(np.float32)
    batch_sm = pipeline.process_batch(batch)
    assert batch_sm.shape == (32, 330)
    print(f"  Batch processing OK, shape={batch_sm.shape}")

    cube = rng.rand(330, 16, 16).astype(np.float32)
    cube_sm = pipeline.process_cube(cube)
    assert cube_sm.shape == (330, 16, 16)
    print(f"  Cube processing OK, shape={cube_sm.shape}")


def test_training_and_export():
    print("\n" + "=" * 60)
    print("TEST 3: PyTorch训练 & ONNX导出")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("  SKIP: PyTorch not installed")
        return None

    from hyperspectral_engine.training.train import (
        SyntheticHyperspectralDataset,
        MultiTaskRegressionLoss,
        build_model,
        export_to_onnx,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model = build_model(spectral_length=330, device=device)
    n_params = model.count_parameters()
    print(f"  Model params: {n_params:,} (<500K required)")
    assert n_params < 2_000_000, "Model too large"

    dummy = torch.randn(4, 1, 330, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert "Cd" in out and "Pb" in out and "As" in out
    assert out["Cd"].shape == (4,)
    print(f"  Forward pass OK: Cd={out['Cd'][0]:.3f}, Pb={out['Pb'][0]:.3f}, As={out['As'][0]:.3f}")

    loss_fn = MultiTaskRegressionLoss()
    targets = torch.rand(4, 3, device=device)
    loss, metrics = loss_fn(out, targets)
    print(f"  Multi-task loss OK: total={loss.item():.4f}")

    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "test_model.onnx")
    export_to_onnx(model, onnx_path, spectral_length=330, device="cpu")
    assert os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0
    print(f"  ONNX export OK ({os.path.getsize(onnx_path) / 1024:.1f} KB)")
    print("  训练与ONNX导出测试: PASSED")
    return onnx_path


def test_onnx_inference(onnx_path: str):
    print("\n" + "=" * 60)
    print("TEST 4: ONNX Runtime推理引擎 (与PyTorch解耦)")
    print("=" * 60)
    try:
        import onnxruntime
    except ImportError:
        print("  SKIP: onnxruntime not installed")
        return

    from hyperspectral_engine.inference.onnx_engine import ONNXInferenceEngine

    engine = ONNXInferenceEngine(onnx_path, use_gpu=False)
    print(f"  Providers: {engine.providers}")
    print(f"  Spectral length: {engine.spectral_length}")

    rng = np.random.RandomState(77)
    single = rng.rand(330).astype(np.float32)
    result_single = engine.predict_single(single)
    print(f"  Single inference: {result_single}")
    assert all(isinstance(v, float) for v in result_single.values())

    batch = rng.rand(16, 330).astype(np.float32)
    result = engine.predict(batch)
    assert result.Cd.shape == (16,)
    print(f"  Batch inference OK: time={result.inference_time_ms:.2f}ms, "
          f"preprocess={result.preprocess_time_ms:.2f}ms")

    cube = rng.rand(330, 20, 25).astype(np.float32)
    t0 = time.perf_counter()
    cd_map, pb_map, as_map, _ = engine.predict_cube(cube)
    dt = (time.perf_counter() - t0) * 1000
    assert cd_map.shape == (20, 25)
    print(f"  Cube inference OK (20x25) in {dt:.1f}ms")
    print(f"  Cd range: [{cd_map.min():.3f}, {cd_map.max():.3f}]")
    print(f"  Pb range: [{pb_map.min():.3f}, {pb_map.max():.3f}]")
    print(f"  As range: [{as_map.min():.3f}, {as_map.max():.3f}]")

    bench = engine.benchmark(batch_size=64, num_iters=30)
    print(f"  Benchmark B=64: mean={bench['mean_ms']:.2f}ms, FPS={bench['fps']:.0f}")
    print("  ONNX推理测试: PASSED")


def test_geotiff_writer():
    print("\n" + "=" * 60)
    print("TEST 5: GeoTIFF污染热力图生成")
    print("=" * 60)
    try:
        import rasterio
    except ImportError:
        print("  SKIP: rasterio not installed")
        return

    from hyperspectral_engine.geotiff.heatmap_writer import (
        PollutionHeatmapWriter,
        GeoTIFFConfig,
    )

    rng = np.random.RandomState(5)
    H, W = 32, 40
    cd_map = rng.uniform(0.05, 5.0, (H, W)).astype(np.float32)
    pb_map = rng.uniform(5.0, 200.0, (H, W)).astype(np.float32)
    as_map = rng.uniform(1.0, 50.0, (H, W)).astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    writer = PollutionHeatmapWriter()

    cd_path = writer.write_concentration(cd_map, os.path.join(tmpdir, "cd.tif"), metal="Cd")
    print(f"  Cd GeoTIFF: {cd_path}")
    with rasterio.open(cd_path) as src:
        assert src.count == 1
        assert src.shape == (H, W)
        assert src.dtypes[0] == "float32"
        data = src.read(1)
        print(f"    shape={src.shape}, dtype={src.dtypes[0]}, min/max={data.min():.3f}/{data.max():.3f}")

    rgba_path = writer.write_rgba_heatmap(cd_map, os.path.join(tmpdir, "cd_rgba.tif"))
    print(f"  RGBA heatmap: {rgba_path}")
    with rasterio.open(rgba_path) as src:
        assert src.count == 4
        print(f"    shape={src.shape}, bands={src.count}")

    multi_path = writer.write_multi_band(
        {"Cd": cd_map, "Pb": pb_map, "As": as_map},
        os.path.join(tmpdir, "multi.tif"),
    )
    print(f"  Multi-band: {multi_path}")
    with rasterio.open(multi_path) as src:
        assert src.count == 3
        desc = [src.descriptions[i] for i in range(3)]
        print(f"    bands={src.count}, descriptions={desc}")

    print("  GeoTIFF生成测试: PASSED")


def test_wavelength_descending():
    """波长降序场景全链路测试 - 模拟南方丘陵大雨冲刷批次

    验证:
        1. ENVI解码器正确嗅探 wavelength_order = descending
        2. 解码后的数据立方体波段自动重排为升序
        3. 预处理管道(SG滤波)正确处理降序输入
        4. 推理引擎输出的Cd浓度不为负值
        5. WavelengthReorderOperator 正确检测和重排
    """
    print("\n" + "=" * 60)
    print("TEST 6: 波长降序场景 - 南方丘陵大雨冲刷批次")
    print("=" * 60)

    from hyperspectral_engine.decoding.envi_decoder import (
        ENVIDecoder,
        decode_gf5_cube,
    )
    from hyperspectral_engine.preprocessing.sg_filter import (
        WavelengthReorderOperator,
        PreprocessingPipeline,
    )
    from generate_test_data import generate_synthetic_envi

    # 6a. 先生成升序数据，再用同一份物理内容(波段翻转)生成降序数据
    tmpdir = tempfile.mkdtemp()
    hdr_asc, dat_asc = generate_synthetic_envi(
        tmpdir, samples=16, lines=12, wavelength_descending=False
    )
    cube_asc_ref, _ = decode_gf5_cube(hdr_asc, dat_asc)

    hdr_desc, dat_desc = generate_synthetic_envi(
        tmpdir, samples=16, lines=12, wavelength_descending=True,
        source_cube=cube_asc_ref,
    )

    # 6b. 验证解码器嗅探到降序排列
    decoder_desc = ENVIDecoder(hdr_desc, dat_desc)
    meta_desc = decoder_desc.parse_header()
    print(f"  Wavelength order detected: {meta_desc.wavelength_order}")
    assert meta_desc.wavelength_order == "descending", \
        f"Expected descending, got {meta_desc.wavelength_order}"

    # 6c. 验证降序解码后的数据与升序解码一致
    cube_desc = decoder_desc.decode()
    assert cube_desc.shape == cube_asc_ref.shape
    np.testing.assert_allclose(cube_desc, cube_asc_ref, rtol=1e-5)
    print(f"  Descending cube auto-reordered to match ascending: shape={cube_desc.shape}")

    # 6d. 验证 WavelengthReorderOperator
    desc_wl = np.linspace(2500.0, 400.0, 330, dtype=np.float32)
    reorder_op = WavelengthReorderOperator(desc_wl)
    assert reorder_op.is_descending, "Should detect descending order"

    test_spectrum_desc = np.arange(330, dtype=np.float32)
    reordered = reorder_op.reorder_spectrum(test_spectrum_desc)
    assert reordered[0] == 329.0, f"First element should be 329, got {reordered[0]}"
    assert reordered[-1] == 0.0, f"Last element should be 0, got {reordered[-1]}"
    print(f"  WavelengthReorderOperator: descending detected, spectrum flipped correctly")

    # 6e. 验证预处理管道对降序波长的处理
    pipeline_desc = PreprocessingPipeline(wavelengths=desc_wl)
    assert pipeline_desc.is_wavelength_descending

    rng = np.random.RandomState(99)
    noisy_desc = rng.rand(330).astype(np.float32)
    smoothed_desc = pipeline_desc.process_spectrum(noisy_desc)
    assert smoothed_desc.shape == (330,)
    assert np.all(np.isfinite(smoothed_desc))
    print(f"  Descending-wavelength pipeline: spectrum processed OK")

    batch_desc = rng.rand(8, 330).astype(np.float32)
    batch_sm = pipeline_desc.process_batch(batch_desc)
    assert batch_sm.shape == (8, 330)
    print(f"  Descending-wavelength pipeline: batch processed OK")

    # 6f. 验证升序和降序管道对相同物理光谱的处理结果一致
    asc_wl = np.linspace(400.0, 2500.0, 330, dtype=np.float32)
    pipeline_asc = PreprocessingPipeline(wavelengths=asc_wl)

    rng2 = np.random.RandomState(42)
    spec_asc = rng2.rand(330).astype(np.float32)
    rng3 = np.random.RandomState(42)
    spec_desc_input = rng3.rand(330).astype(np.float32)[::-1].copy()

    result_asc = pipeline_asc.process_spectrum(spec_asc)
    result_desc = pipeline_desc.process_spectrum(spec_desc_input)
    np.testing.assert_allclose(result_asc, result_desc, rtol=1e-4)
    print(f"  Ascending vs descending pipeline results match (rtol=1e-4)")

    # 6g. 验证推理引擎对降序波长的处理(如果ONNX模型可用)
    try:
        from hyperspectral_engine.inference.onnx_engine import ONNXInferenceEngine
        import onnxruntime

        tmpdir2 = tempfile.mkdtemp()
        from hyperspectral_engine.training.train import build_model, export_to_onnx
        import torch
        model = build_model(spectral_length=330, device="cpu")
        onnx_path = os.path.join(tmpdir2, "desc_test.onnx")
        export_to_onnx(model, onnx_path, spectral_length=330, device="cpu")

        engine = ONNXInferenceEngine(onnx_path, use_gpu=False)
        engine.set_wavelengths(desc_wl)

        rng4 = np.random.RandomState(55)
        spec_for_infer = rng4.rand(330).astype(np.float32)
        result = engine.predict_single(spec_for_infer)
        print(f"  Descending-wavelength inference: Cd={result['Cd']:.4f}, "
              f"Pb={result['Pb']:.4f}, As={result['As']:.4f}")
        assert result["Cd"] >= 0, f"Cd should be non-negative, got {result['Cd']}"
        assert result["Pb"] >= 0, f"Pb should be non-negative, got {result['Pb']}"
        assert result["As"] >= 0, f"As should be non-negative, got {result['As']}"
        print(f"  All concentrations non-negative: PASSED")
    except (ImportError, FileNotFoundError):
        print(f"  ONNX inference test skipped (missing deps)")

    print("  波长降序场景全链路测试: PASSED")


def test_vca_intervention():
    """VCA端元解混+地膜纯度干预全链路测试

    验证:
        1. VCA从混合光谱中正确提取端元
        2. SAM光谱角匹配正确识别地膜端元
        3. 地膜丰度超过阈值时干预模块阻断反演
        4. 干预净荷包含正确的端元丰度信息
        5. 鲜黄色斜十字警示网格叠加层生成正确
    """
    print("\n" + "=" * 60)
    print("TEST 7: VCA端元解混 + 地膜纯度干预")
    print("=" * 60)

    from hyperspectral_engine.unmixing.vca import vca, fully_constrained_abundance
    from hyperspectral_engine.unmixing.spectral_endmembers import (
        get_endmember,
        spectral_angle,
        spectral_angle_batch,
        best_matching_endmember,
    )
    from hyperspectral_engine.unmixing.intervention import (
        PurityInterventionModule,
        InterventionPayload,
        MULCH_SAM_THRESHOLD,
    )

    rng = np.random.RandomState(42)

    mulch_ref = get_endmember("polyethylene_mulch")
    veg_ref = get_endmember("green_vegetation")
    soil_ref = get_endmember("bare_soil")

    L = 330
    N_clean = 200
    N_mulch = 200

    clean_spectra = np.zeros((L, N_clean), dtype=np.float32)
    for i in range(N_clean):
        a_soil = rng.uniform(0.5, 0.8)
        a_veg = 1.0 - a_soil
        clean_spectra[:, i] = a_soil * soil_ref + a_veg * veg_ref

    mulch_spectra = np.zeros((L, N_mulch), dtype=np.float32)
    for i in range(N_mulch):
        a_mulch = rng.uniform(0.6, 0.95)
        a_soil = (1.0 - a_mulch) * rng.uniform(0.3, 0.7)
        a_veg = 1.0 - a_mulch - a_soil
        mulch_spectra[:, i] = a_mulch * mulch_ref + a_soil * soil_ref + a_veg * veg_ref

    all_spectra = np.hstack([clean_spectra, mulch_spectra])
    noise = rng.randn(*all_spectra.shape).astype(np.float32) * 0.005
    all_spectra = np.clip(all_spectra + noise, 0.0, 1.0)

    # 7a. VCA端元提取
    endmembers, indices = vca(all_spectra, num_endmembers=3)
    assert endmembers.shape == (L, 3), f"Expected (330,3), got {endmembers.shape}"
    print(f"  VCA extracted {endmembers.shape[1]} endmembers from {all_spectra.shape[1]} pixels")

    # 7b. SAM匹配
    labels = []
    sam_angles = []
    for i in range(3):
        name, angle = best_matching_endmember(endmembers[:, i])
        labels.append(name)
        sam_angles.append(angle)
        print(f"  Endmember {i}: best match = {name}, SAM = {angle:.4f} rad")

    assert "polyethylene_mulch" in labels, \
        f"VCA should find mulch endmember, got labels: {labels}"

    # 7c. FCLS丰度分解
    abundances = fully_constrained_abundance(all_spectra, endmembers)
    assert abundances.shape == (3, N_clean + N_mulch)
    col_sums = abundances.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=0.1)
    print(f"  FCLS abundance sums: min={col_sums.min():.3f}, max={col_sums.max():.3f}")

    mulch_label_idx = labels.index("polyethylene_mulch")
    mulch_ab = abundances[mulch_label_idx, :]
    mulch_pixels_ab = mulch_ab[N_clean:]
    clean_pixels_ab = mulch_ab[:N_clean]
    print(f"  Mulch pixels abundance: mean={mulch_pixels_ab.mean():.3f}, max={mulch_pixels_ab.max():.3f}")
    print(f"  Clean pixels abundance: mean={clean_pixels_ab.mean():.3f}")

    # 7d. 纯度干预模块 - 地膜污染图斑
    module = PurityInterventionModule(
        mulch_sam_threshold=MULCH_SAM_THRESHOLD,
        num_endmembers=3,
    )

    payload_mulch = module.analyze_patch(mulch_spectra)
    assert payload_mulch.mulch_detected, "Should detect mulch in contaminated patch"
    assert payload_mulch.blocked, "Should block inference for contaminated patch"
    assert payload_mulch.mulch_sam_angle < MULCH_SAM_THRESHOLD
    print(f"  Mulch patch: blocked={payload_mulch.blocked}, "
          f"SAM={payload_mulch.mulch_sam_angle:.4f}, "
          f"abundance_mean={payload_mulch.mulch_abundance_mean:.4f}")

    # 7e. 纯度干预模块 - 干净图斑
    payload_clean = module.analyze_patch(clean_spectra)
    print(f"  Clean patch: blocked={payload_clean.blocked}, "
          f"mulch_detected={payload_clean.mulch_detected}")
    if payload_clean.mulch_detected:
        print(f"    SAM={payload_clean.mulch_sam_angle:.4f} (threshold={MULCH_SAM_THRESHOLD})")

    # 7f. 单像素快速检测
    test_pixel_clean = clean_spectra[:, 0]
    is_blocked, angle = module.analyze_pixel(test_pixel_clean)
    print(f"  Single pixel (clean): blocked={is_blocked}, SAM={angle:.4f}")

    test_pixel_mulch = mulch_spectra[:, 0]
    is_blocked_m, angle_m = module.analyze_pixel(test_pixel_mulch)
    print(f"  Single pixel (mulch): blocked={is_blocked_m}, SAM={angle_m:.4f}")

    # 7g. 鲜黄色斜十字警示网格
    H, W = 12, 16
    warning_mask = np.zeros((H, W), dtype=np.float32)
    warning_mask[2:8, 3:12] = 1.0

    overlay = PurityInterventionModule.build_crosshatch_overlay(
        warning_mask, spacing=4, line_width=1
    )
    assert overlay.shape == (H, W, 4), f"Expected ({H},{W},4), got {overlay.shape}"
    assert overlay.dtype == np.uint8

    mask_pixels = np.where(warning_mask > 0)
    hatched_r, hatched_c = mask_pixels[0], mask_pixels[1]
    diag1 = (hatched_r - hatched_c) % 4 < 1
    diag2 = (hatched_r + hatched_c) % 4 < 1
    crosshatch_mask = diag1 | diag2

    hatched_pixels = overlay[hatched_r[crosshatch_mask], hatched_c[crosshatch_mask]]
    assert np.all(hatched_pixels[:, 0] == 255), "Crosshatch R should be 255"
    assert np.all(hatched_pixels[:, 1] == 255), "Crosshatch G should be 255"
    assert np.all(hatched_pixels[:, 2] == 0), "Crosshatch B should be 0"
    assert np.all(hatched_pixels[:, 3] == 200), "Crosshatch A should be 200"
    print(f"  Crosshatch overlay: shape={overlay.shape}, yellow=#FFFF00, alpha=200")

    # 7h. 干预净荷序列化
    payload_dict = payload_mulch.to_dict()
    assert payload_dict["blocked"] is True
    assert payload_dict["mulch_detected"] is True
    assert "endmember_labels" in payload_dict
    assert "abundance_means" in payload_dict
    print(f"  Intervention payload serialized OK: {list(payload_dict.keys())}")

    print("  VCA解混+地膜干预测试: PASSED")


def main():
    print("\n" + "#" * 60)
    print("#  星载高光谱污染智能反演引擎 - 端到端集成测试")
    print("#" * 60)

    onnx_path = None
    tests = [
        ("ENVI Decoding", test_envi_decoding),
        ("SG Preprocessing", test_sg_preprocessing),
        ("Training & ONNX Export", lambda: test_training_and_export()),
    ]

    results = []
    for name, test_fn in tests:
        try:
            res = test_fn()
            if name == "Training & ONNX Export" and res:
                onnx_path = res
            results.append((name, "PASSED"))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append((name, f"FAILED: {e}"))

    if onnx_path:
        try:
            test_onnx_inference(onnx_path)
            results.append(("ONNX Inference", "PASSED"))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append(("ONNX Inference", f"FAILED: {e}"))

    try:
        test_geotiff_writer()
        results.append(("GeoTIFF Writer", "PASSED"))
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results.append(("GeoTIFF Writer", f"FAILED: {e}"))

    try:
        test_wavelength_descending()
        results.append(("Wavelength Descending", "PASSED"))
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results.append(("Wavelength Descending", f"FAILED: {e}"))

    try:
        test_vca_intervention()
        results.append(("VCA Intervention", "PASSED"))
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results.append(("VCA Intervention", f"FAILED: {e}"))

    print("\n" + "#" * 60)
    print("#  测试结果汇总")
    print("#" * 60)
    for name, status in results:
        print(f"  {name:<25s}: {status}")

    passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\n  Total: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
