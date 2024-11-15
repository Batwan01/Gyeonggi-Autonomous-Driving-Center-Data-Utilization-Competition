import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def weight_pruning(model, amount=0.3):
    """Apply L1 unstructured pruning to the model's convolutional layers"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')

def channel_pruning(model, amount=0.3):
    """Apply channel-wise pruning based on L1-norm"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                # Skip if this is a depthwise conv or has special requirements
                if module.groups > 1:
                    continue
                    
                # Calculate L1-norm for each input channel
                channel_norms = torch.norm(module.weight.data, p=1, dim=(0,2,3))
                num_channels = len(channel_norms)
                
                # Ensure we don't prune too many channels
                num_channels_to_prune = min(int(num_channels * amount), num_channels - 1)
                if num_channels_to_prune <= 0:
                    continue
                
                # Find channels to keep
                threshold = torch.sort(channel_norms)[0][num_channels_to_prune]
                mask = channel_norms > threshold
                
                # Ensure we keep at least one channel
                if not torch.any(mask):
                    mask[torch.argmax(channel_norms)] = True
                
                # Apply mask
                module.weight.data = module.weight.data[:, mask, :, :]
                
                # Update in_channels
                module.in_channels = int(torch.sum(mask))
                
                # Handle bias if it exists
                if module.bias is not None and mask.shape[0] == module.bias.shape[0]:
                    module.bias.data = module.bias.data[mask]
                
                print(f"Pruned {name}: {num_channels} -> {int(torch.sum(mask))} channels")
                
            except Exception as e:
                print(f"Skipping layer {name} due to error: {str(e)}")
                continue

def filter_pruning(model, amount=0.3):
    """Apply filter pruning based on L2-norm, preserving the output layer"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Skip the final detection layer
            if 'dfl' in name or name.endswith('.cv3'):  # YOLO의 마지막 detection layer들
                continue
                
            # Calculate L2-norm for each filter
            filter_norms = torch.norm(module.weight.data, p=2, dim=(1,2,3))
            num_filters = len(filter_norms)
            num_filters_to_prune = int(num_filters * amount)
            
            # Find filters to keep
            threshold = torch.sort(filter_norms)[0][num_filters_to_prune]
            mask = filter_norms > threshold
            
            # Apply mask
            module.weight.data = module.weight.data[mask, :, :, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[mask]
            
            # Update module parameters
            module.out_channels = int(torch.sum(mask))

def print_model_structure(model, title="Model Structure"):
    """Print simplified model structure focusing on parameter changes"""
    print(f"\n{title}")
    print("-" * 50)
    print(f"{'Layer':<20} {'Channels (in/out)':<20} {'Parameters':<15}")
    print("-" * 50)
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"{name[-20:]:<20} {f'{module.in_channels}/{module.out_channels}':<20} {params:,}")
    print("-" * 50)
    print(f"Total parameters: {total_params:,}")

def train_with_pruning(amount):
    # 1. 학습된 모델 로드
    model = YOLO('gyeonggi_AD_competition/yolo11s_fold3/weights/best.pt')
    
    # 프루닝 전 모델 구조 출력
    print_model_structure(model.model, "Before Pruning")

    # 2. 프루닝 적용
    if pruned_type == 'weight':
        weight_pruning(model.model, amount=amount)
    elif pruned_type == 'channel':
        channel_pruning(model.model, amount=amount)
    elif pruned_type == 'filter':
        filter_pruning(model.model, amount=amount)
    
    # 프루닝 후 모델 구조 출력
    print_model_structure(model.model, "After Pruning")

    # 3. 프루닝된 모델 fine-tuning
    model.train(data='../data/yaml/dataset_fold3.yaml',
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                project='gyeonggi_AD_competition',
                name=f'yolo11s_fold3_{pruned_type}{amount}',
                # model=model
                )
    
    print("\nAfter Training:")
    print_model_structure(model.model)

    # 4. 프루닝된 모델 저장
    # model.save(f'yolo11s_fold3_{pruned_type}{amount}')

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    epochs = 100
    imgsz = 640
    batch_size = 32
    pruned_type = 'channel'
    # 여러 amount 값으로 실험
    pruning_amounts = [0.1, 0.2, 0.3]
    
    for amount in pruning_amounts:
        print(f"\nStarting training with pruning amount: {amount}")
        train_with_pruning(amount)
        print(f"Completed training with pruning amount: {amount}")