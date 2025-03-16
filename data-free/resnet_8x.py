import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize_coefs=None, normalize=False):
        super(ResNet, self).__init__()

        if normalize_coefs is not None:
            self.mean, self.std = normalize_coefs

        self.normalize = normalize

        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):

        if self.normalize:
            # Normalize according to the training data normalization statistics
            x -= self.mean
            x /= self.std

        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature
 
 
def ResNet18_8x(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def ResNet34_8x(num_classes=10, normalize_coefs=None, normalize=False):   
    return ResNet(BasicBlock, [3,4,6,3], num_classes, normalize_coefs=normalize_coefs, normalize=normalize)

def ResNet50_8x(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101_8x(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152_8x(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

import torch.nn as nn
import torch.nn.functional as F



class EarlyExitBlock(nn.Module):
    def __init__(self, planes, num_classes):
        super(EarlyExitBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(planes, num_classes*10)
        self.fc2 = nn.Linear(num_classes*10, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        probs = self.softmax(x)
        return x, probs

class ResNet_EarlyExit(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, exit_thresholds=[0.9, 0.92, 0.94, 0.96], normalize_coefs=None, normalize=False):
        super(ResNet_EarlyExit, self).__init__()
      
        if normalize_coefs is not None:
            self.mean, self.std = normalize_coefs
        self.normalize = normalize
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.early_exit_blocks = nn.ModuleList([
            EarlyExitBlock(64 * block.expansion, num_classes),
            EarlyExitBlock(128 * block.expansion, num_classes),
            EarlyExitBlock(256 * block.expansion, num_classes),
            EarlyExitBlock(512 * block.expansion, num_classes)
        ])
        self.exit_thresholds = exit_thresholds
        self.use_early_exit = True
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, out_feature=False, return_all_outputs=False):

        if self.normalize:
            # Normalize according to the training data normalization statistics
            x -= self.mean
            x /= self.std
        x = F.relu(self.bn1(self.conv1(x)))

        exit_indices = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)

        outputs = torch.zeros((x.size(0), self.num_classes), device=x.device)

        not_exited = torch.ones(x.size(0), dtype=torch.bool, device=x.device)  
        
        early_exit_outputs = []
        if self.use_early_exit and return_all_outputs:
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)  
                if i < len(self.early_exit_blocks):
                
                    early_exit_output, probs = self.early_exit_blocks[i](x)
                    early_exit_outputs.append(early_exit_output)  
                    not_exited = (exit_indices == -1)  
                    if not_exited.any():
                        max_probs, _ = torch.max(probs, dim=1)
                        exit_now = max_probs > self.exit_thresholds[i]  
                        exit_now = exit_now & not_exited 
                        exit_indices[exit_now] = i  
            out = F.avg_pool2d(x, 4)
            feature = out.view(out.size(0), -1)
            out = self.linear(feature)
            early_exit_outputs.append(out) 
            exit_indices[exit_indices == -1] = len(self.early_exit_blocks)
            return early_exit_outputs, exit_indices

                 
        if self.use_early_exit and not return_all_outputs:
            for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                x = layer(x)
                if i < len(self.early_exit_blocks):
                    early_exit_result = self.early_exit_blocks[i](x)
                    early_exit_output = early_exit_result[0]  
                    if not_exited.any():
                        valid_early_exit_output = early_exit_output[not_exited]
                        outputs[not_exited] = valid_early_exit_output
                        max_probs, _ = torch.max(valid_early_exit_output, dim=1)
                        exit_now = max_probs > self.exit_thresholds[i]
                        exit_indices[not_exited] = torch.where(exit_now, i, exit_indices[not_exited])
                        not_exited &= ~exit_now
            if not_exited.any():
                x_remaining = x[not_exited]
                x_remaining = F.avg_pool2d(x_remaining, 4)
                feature = x_remaining.view(x_remaining.size(0), -1)
                final_output = self.linear(feature)
                outputs[not_exited] = final_output
                exit_indices[not_exited] = len(self.early_exit_blocks)
            return outputs, exit_indices     
        elif not self.use_early_exit:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                x = layer(x)
            x = F.avg_pool2d(x, 4)
            feature = x.view(x.size(0), -1)
            final_output = self.linear(feature)
            exit_indices[not_exited] = 4
            return final_output, exit_indices    
    def enable_early_exit(self):
        self.use_early_exit = True
    def disable_early_exit(self):
        self.use_early_exit = False


def ResNet34_8x_EarlyExit(num_classes=10, exit_thresholds=[0.9, 0.92, 0.94, 0.96], normalize_coefs=None, normalize=False):  
    return ResNet_EarlyExit(BasicBlock, [3,4,6,3], num_classes, exit_thresholds, normalize_coefs=normalize_coefs, normalize=normalize)


