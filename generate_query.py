import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import os

def generate_bulk_properties(
    onnx_file="new_fashion_mnist_diamond_relu.onnx",
    save_dir="./properties",
    data_root="./data",
    epsilons=None,
    target_image_count=5,
    clear_existing=True,
):
    # 모델 로드 (원본을 맞히는지 테스트하기 위함)
    if not os.path.exists(onnx_file):
        print(f"Error: {onnx_file} 파일이 없습니다.")
        return 0
        
    session = ort.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name

    # test data 로드
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    
    os.makedirs(save_dir, exist_ok=True)

    if clear_existing:
        for filename in os.listdir(save_dir):
            if filename.endswith(".txt"):
                os.remove(os.path.join(save_dir, filename))
    
    if epsilons is None:
        epsilons = [0.001, 0.003, 0.005, 0.045]
    
    # 테스트할 '정답 맞힌' 이미지 목표 개수
    correct_images_found = 0
    files_written = 0
    img_idx = 0
    
    print("🔍 모델이 정답을 맞힌 이미지를 검색하며 파일을 생성합니다...")

    # 정답 맞힌 이미지를 5장 찾을 때까지 데이터셋을 순회합니다.
    while correct_images_found < target_image_count and img_idx < len(dataset):
        image_tensor, true_label = dataset[img_idx]
        
        # 모델이 원본 이미지를 맞췄는지 확인
        input_data = image_tensor.unsqueeze(0).numpy() # (1, 1, 28, 28) 형태
        outputs = session.run(None, {input_name: input_data})
        pred_label = np.argmax(outputs[0])
        
        # 정답을 틀린 이미지는 스킵
        if pred_label != true_label:
            img_idx += 1
            continue
            
        print(f"[{correct_images_found+1}/{target_image_count}] 올바르게 분류된 이미지 발견! (Index: {img_idx}, Label: {true_label})")
        
        # property 파일 생성 로직
        image_flat = image_tensor.numpy().flatten()
        
        for target_class in range(10):
            if target_class == true_label:
                continue
                
            for eps in epsilons:
                filename = os.path.join(
                    save_dir, f"image{img_idx}_target{target_class}_epsilon{eps}.txt"
                )
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("// Input variables\n")
                    for i, pixel_val in enumerate(image_flat):
                        lower = max(0.0, float(pixel_val) - eps) 
                        # 기존 방식은 음수 값이 존재했으나 이미지 data에서 음수는 존재할 수 없으므로 0 이상으로 clip
                        upper = min(1.0, float(pixel_val) + eps)
                        # 기존 방식은 1 이상의 값이 존재할 수 있었으나 이미지에서 1보다 큰 값은 없도록 전처리 했기 때문에 1 이하로 clip
                        f.write(f"x{i} >= {lower}\n")
                        f.write(f"x{i} <= {upper}\n")
                        
                    f.write("// Output variables\n")
                    f.write(f"+y{true_label} -y{target_class} <= 0\n")
                files_written += 1
        
        correct_images_found += 1
        img_idx += 1
                            
    print(f"\n✅ '{save_dir}' 폴더에 총 {files_written}개의 텍스트 파일 생성이 완료되었습니다.")
    return files_written

if __name__ == "__main__":
    generate_bulk_properties()
