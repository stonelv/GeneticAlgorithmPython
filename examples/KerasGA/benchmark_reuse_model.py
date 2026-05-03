import time
import numpy
import tensorflow.keras
import pygad.kerasga


def create_test_model(input_shape=(3,), output_units=1):
    input_layer = tensorflow.keras.layers.Input(input_shape)
    dense_layer1 = tensorflow.keras.layers.Dense(32, activation="relu")(input_layer)
    dense_layer2 = tensorflow.keras.layers.Dense(16, activation="relu")(dense_layer1)
    output_layer = tensorflow.keras.layers.Dense(output_units, activation="linear")(dense_layer2)
    return tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)


def create_test_data(num_samples=1000, input_shape=(3,)):
    numpy.random.seed(42)
    data_inputs = numpy.random.rand(num_samples, *input_shape)
    data_outputs = numpy.random.rand(num_samples, 1)
    return data_inputs, data_outputs


def benchmark_predict(model, solution, data, num_calls=100, reuse_mode=None, reuse_model_instance=None):
    pygad.kerasga.clear_model_cache()
    
    times = []
    predictions_list = []
    
    for i in range(num_calls):
        start = time.perf_counter()
        
        if reuse_model_instance is not None:
            pred = pygad.kerasga.predict(
                model=model,
                solution=solution,
                data=data,
                reuse_model=reuse_model_instance
            )
        else:
            pred = pygad.kerasga.predict(
                model=model,
                solution=solution,
                data=data,
                reuse_model=reuse_mode
            )
        
        end = time.perf_counter()
        times.append(end - start)
        predictions_list.append(pred)
    
    return times, predictions_list


def main():
    print("=" * 60)
    print("KerasGA predict() 缓存复用基准测试")
    print("=" * 60)
    
    numpy.random.seed(42)
    
    print("\n[1/4] 创建测试模型和数据...")
    model = create_test_model()
    data_inputs, data_outputs = create_test_data(num_samples=100)
    
    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)
    solution = keras_ga.population_weights[0]
    
    NUM_CALLS = 50
    
    print(f"\n[2/4] 开始基准测试 (每个模式调用 {NUM_CALLS} 次)...")
    
    print("\n  模式 A: reuse_model=None (默认 - 每次 clone_model)")
    times_none, preds_none = benchmark_predict(
        model, solution, data_inputs, 
        num_calls=NUM_CALLS, reuse_mode=None
    )
    avg_time_none = numpy.mean(times_none)
    total_time_none = numpy.sum(times_none)
    print(f"    平均单次: {avg_time_none*1000:.2f} ms")
    print(f"    总计: {total_time_none:.4f} s")
    
    print("\n  模式 B: reuse_model=True (自动缓存克隆模型)")
    times_cache, preds_cache = benchmark_predict(
        model, solution, data_inputs, 
        num_calls=NUM_CALLS, reuse_mode=True
    )
    avg_time_cache = numpy.mean(times_cache)
    total_time_cache = numpy.sum(times_cache)
    print(f"    平均单次: {avg_time_cache*1000:.2f} ms")
    print(f"    总计: {total_time_cache:.4f} s")
    
    print("\n  模式 C: reuse_model=user_model (用户提供复用模型)")
    user_reuse_model = tensorflow.keras.models.clone_model(model)
    times_user, preds_user = benchmark_predict(
        model, solution, data_inputs, 
        num_calls=NUM_CALLS, reuse_model_instance=user_reuse_model
    )
    avg_time_user = numpy.mean(times_user)
    total_time_user = numpy.sum(times_user)
    print(f"    平均单次: {avg_time_user*1000:.2f} ms")
    print(f"    总计: {total_time_user:.4f} s")
    
    print("\n[3/4] 性能对比...")
    speedup_cache = avg_time_none / avg_time_cache if avg_time_cache > 0 else float('inf')
    speedup_user = avg_time_none / avg_time_user if avg_time_user > 0 else float('inf')
    
    print(f"\n  模式 B (自动缓存) 相对模式 A 加速: {speedup_cache:.2f}x")
    print(f"  模式 C (用户模型) 相对模式 A 加速: {speedup_user:.2f}x")
    
    print("\n[4/4] 验证输出一致性...")
    
    all_close_cache = numpy.allclose(preds_none[0], preds_cache[0])
    max_diff_cache = numpy.max(numpy.abs(preds_none[0] - preds_cache[0]))
    
    all_close_user = numpy.allclose(preds_none[0], preds_user[0])
    max_diff_user = numpy.max(numpy.abs(preds_none[0] - preds_user[0]))
    
    print(f"\n  模式 A 与 模式 B 输出一致: {all_close_cache}")
    print(f"  最大差异: {max_diff_cache:.2e}")
    
    print(f"\n  模式 A 与 模式 C 输出一致: {all_close_user}")
    print(f"  最大差异: {max_diff_user:.2e}")
    
    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)
    
    print("\n详细统计:")
    print(f"{'模式':<30} {'平均(ms)':<12} {'总计(s)':<10} {'加速比':<10}")
    print("-" * 60)
    print(f"{'A: reuse_model=None (每次clone)':<30} {avg_time_none*1000:<12.2f} {total_time_none:<10.4f} {'1.00x':<10}")
    print(f"{'B: reuse_model=True (自动缓存)':<30} {avg_time_cache*1000:<12.2f} {total_time_cache:<10.4f} {speedup_cache:.2f}x")
    print(f"{'C: reuse_model=实例 (用户提供)':<30} {avg_time_user*1000:<12.2f} {total_time_user:<10.4f} {speedup_user:.2f}x")
    
    print("\n一致性检查结果:")
    if all_close_cache and all_close_user:
        print("  ✓ 所有模式输出完全一致")
    else:
        print("  ✗ 警告: 存在输出不一致的情况")
    
    pygad.kerasga.clear_model_cache()
    return all_close_cache and all_close_user


if __name__ == "__main__":
    main()
