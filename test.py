import numpy as np
import time
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import re
import warnings
from collections import defaultdict
warnings.filterwarnings(
    "ignore",
    message="Tensorflow parser is unavailable because tensorflow package is not installed",
    category=UserWarning,
)
from maraboupy import Marabou
from generate_query import generate_bulk_properties
from model import train_and_export


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_FILE = os.path.join(BASE_DIR, "new_fashion_mnist_diamond_relu.onnx")
PROPERTY_DIR = os.path.join(BASE_DIR, "properties")
REPORT_FILE = os.path.join(BASE_DIR, "verification_report_aggregated.txt")
SAT_IMAGE_FILE = os.path.join(BASE_DIR, "sat_images.png")
DATA_ROOT = os.path.join(BASE_DIR, "data")
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
SEED = 42
TARGET_IMAGE_COUNT = 5
EPSILONS = [0.001, 0.003, 0.005, 0.045]
TIMEOUT_SECONDS = 180


def visualize_sat_images(sat_results, output_file="sat_images.png"):
    if not sat_results:
        print("SAT 결과가 나온 이미지가 없어 시각화할 항목이 없습니다.")
        return

    try:
        import matplotlib.pyplot as plt
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        print(f"시각화에 필요한 패키지를 불러오지 못했습니다: {exc}")
        return

    # test data 로드
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=False, transform=transform
    )

    # 5개의 이미지를 분석하므로 최대 5개가 필요함
    cols = min(5, len(sat_results))
    rows = int(np.ceil(len(sat_results) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 4.2 * rows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(sat_results):]:
        ax.axis("off")

    for ax, result in zip(axes, sat_results):
        image_tensor, true_label = dataset[result["image_index"]]
        image = image_tensor.squeeze(0).numpy()
        class_name = FASHION_MNIST_CLASSES[true_label]
        target = result["target"]
        target_label = (
            FASHION_MNIST_CLASSES[target]
            if target is not None and 0 <= target < len(FASHION_MNIST_CLASSES)
            else "Unknown"
        )

        # 이미지 시각화
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"True: {true_label} ({class_name})\n"
            f"{result['image_name']}, eps={result['epsilon']}\n"
            f"Target: {target} ({target_label})",
            fontsize=9,
            pad=10,
        )
        ax.axis("off")

    fig.savefig(output_file, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"SAT 이미지 시각화가 '{output_file}'에 저장되었습니다.")


def run_verification(onnx_file, prop_dir, report_file, timeout_seconds, sat_image_file):
    
    if not os.path.exists(prop_dir):
        print(f"Error: {prop_dir} 폴더가 없습니다.")
        return

    prop_files = sorted([f for f in os.listdir(prop_dir) if f.endswith('.txt')])
    if not prop_files:
        raise RuntimeError(f"{prop_dir} 폴더에 검증할 .txt property 파일이 없습니다.")
    
    # Timeout 180초 설정(너무 오래 걸리는 것 막기 위해)
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout_seconds)

    # 파일들을 '이미지'와 '입실론(eps)' 기준으로 그룹화 (9개씩 묶기 위함)
    # groups['image1']['0.005'] = [파일 리스트...]
    groups = defaultdict(lambda: defaultdict(list))
    sat_results = []
    
    for f in prop_files:
        # 파일명에서 image 번호와 epsilon 값 추출
        match = re.match(r"(image\d+)_target\d+_epsilon([\d\.]+)\.txt", f)
        if match:
            img_key = match.group(1)
            eps_key = match.group(2)
            groups[img_key][eps_key].append(f)

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("================================================================================\n")
        report.write(f"Marabou Verification Aggregated Report - {time.ctime()}\n")
        report.write(f"Model: {onnx_file} | Timeout per query: {timeout_seconds}s\n")
        report.write("================================================================================\n")
        report.write(f"{'Image':<10} | {'Epsilon':<10} | {'Final Result':<12} | {'Total Time(s)':<15}\n")
        report.write("-" * 56 + "\n")

        print(f"총 {len(prop_files)}개의 쿼리를 이미지별로 종합하여 검증합니다.\n")

        # 이미지를 숫자에 맞게 정렬 (image1, image2, ..., image5)
        sorted_images = sorted(groups.keys(), key=lambda x: int(x.replace('image', '')))

        for img_name in sorted_images:
            sorted_eps = sorted(groups[img_name].keys(), key=lambda x: float(x))

            for eps_val in sorted_eps:
                files = sorted(
                    groups[img_name][eps_val],
                    key=lambda name: int(re.search(r"_target(\d+)_", name).group(1))
                )
                
                overall_result = "UNSAT"
                total_time = 0.0
                any_timeout = False

                print(f"[{img_name} - eps:{eps_val}] 검증 시작 (오답 타겟 {len(files)}개 탐색 중)...")

                for filename in files:
                    prop_path = os.path.join(prop_dir, filename)
                    
                    network = Marabou.read_onnx(onnx_file)
                    ipq = network.getInputQuery()
                    
                    start_time = time.time()
                    
                    results = Marabou.solve_query(ipq, options=options, propertyFilename=prop_path, verbose=False)
                    exitCode, vals, stats = results[0], results[1], results[2]
                    
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    
                    if stats.hasTimedOut():
                        any_timeout = True

                    if exitCode == "sat":
                        overall_result = "SAT"
                        target_match = re.search(r"_target(\d+)_", filename)
                        sat_results.append({
                            "image_name": img_name,
                            "image_index": int(img_name.replace("image", "")),
                            "epsilon": eps_val,
                            "target": int(target_match.group(1)) if target_match else None,
                        })
                        break 

                # 9개를 다 돌았는데 SAT는 못 찾았지만, 중간에 타임아웃이 있었다면 완벽한 방어(UNSAT)가 아님
                if overall_result != "SAT" and any_timeout:
                    overall_result = "TIMEOUT"
                
                line = f"{img_name:<10} | {eps_val:<10} | {overall_result:<12} | {total_time:<15.2f}\n"
                report.write(line)
                report.flush()
                
                print(f" => 완료: {img_name} (eps:{eps_val}) | 결과: {overall_result} | 합산 시간: {total_time:.2f}s\n")

        report.write("-" * 56 + "\n")
        report.write("Aggregated Verification Completed.\n")

    print(f"모든 종합 결과가 '{report_file}'에 저장되었습니다.")
    visualize_sat_images(sat_results, output_file=sat_image_file)

def main():
    print("1/3 모델 학습 및 ONNX export를 시작합니다.")
    train_and_export(
        onnx_filename=ONNX_FILE,
        epochs=EPOCHS,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )

    print("\n2/3 Marabou property query 생성을 시작합니다.")
    total_files = generate_bulk_properties(
        onnx_file=ONNX_FILE,
        save_dir=PROPERTY_DIR,
        data_root=DATA_ROOT,
        epsilons=EPSILONS,
        target_image_count=TARGET_IMAGE_COUNT,
        clear_existing=True,
    )
    if total_files == 0:
        raise RuntimeError("생성된 property query가 없습니다.")

    print("\n3/3 Marabou 검증을 시작합니다.")
    run_verification(
        onnx_file=ONNX_FILE,
        prop_dir=PROPERTY_DIR,
        report_file=REPORT_FILE,
        timeout_seconds=TIMEOUT_SECONDS,
        sat_image_file=SAT_IMAGE_FILE,
    )

if __name__ == "__main__":
    main()
