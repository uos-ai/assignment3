import torchvision
import torchvision.transforms as transforms
import os

def generate_bulk_properties():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    save_dir = "./properties"
    os.makedirs(save_dir, exist_ok=True)
    
    epsilons = [0.005, 0.05, 0.1]
    image_indices = [0, 1, 2] # 테스트할 이미지 3장 (index 0, 1, 2)
    
    for img_idx in image_indices:
        image_tensor, true_label = dataset[img_idx]
        image_flat = image_tensor.numpy().flatten()
        
        # 0부터 9까지 타겟 클래스를 설정
        for target_class in range(10):
            # 오답으로 유도하는 것이 목적이므로, 타겟이 실제 정답인 경우는 생성하지 않고 건너뜁니다.
            if target_class == true_label:
                continue
                
            for eps in epsilons:
                filename = f"{save_dir}/image{img_idx+1}_target{target_class}_epsilon{eps}.txt"
                
                with open(filename, 'w') as f:
                    f.write("// Input variables\n")
                    for i, pixel_val in enumerate(image_flat):
                        lower = max(0.0, float(pixel_val) - eps)
                        upper = min(1.0, float(pixel_val) + eps)
                        f.write(f"x{i} >= {lower}\n")
                        f.write(f"x{i} <= {upper}\n")
                        
                    f.write("// Output variables\n")
                    
                    # 벤치마크 원본과 동일한 로직: 
                    # "타겟 클래스(target_class)가 다른 모든 클래스(i)보다 크거나 같아야 한다"
                    for i in range(10):
                        if i != target_class:
                            f.write(f"+y{i} -y{target_class} <= 0\n")
                            
    print(f"'{save_dir}' 폴더에 벤치마크 규격과 동일한 텍스트 파일 생성이 완료되었습니다.")

if __name__ == "__main__":
    generate_bulk_properties()