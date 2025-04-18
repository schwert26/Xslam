#include "XFeat.h"
#include <tuple>

namespace ORB_SLAM3
{
    /////////////////////////////  MODEL IMPLEMENTATION  ////////////////////////////////////
    MBABlockImpl::MBABlockImpl(int reduction) {
    // 初始化卷积层
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 1, 7)  .padding(3).bias(false)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 1, 3)  .padding(1).bias(false)));
    // 初始化激活函数
    sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
    // 参数初始化
    _initialize_weights();
    }

    void MBABlockImpl::_initialize_weights() {
        // 对卷积层应用Kaiming初始化
        torch::nn::init::kaiming_normal_(conv1->weight, 0.0, torch::kFanOut, torch::kReLU);
        torch::nn::init::kaiming_normal_(conv2->weight, 0.0, torch::kFanOut, torch::kReLU);
    }

    torch::Tensor MBABlockImpl::forward(torch::Tensor x) {
        // 平均池化和最大池化
        auto avg_pool = torch::mean(x, 1, /*keepdim=*/true);
        auto max_pool = std::get<0>(torch::max(x, 1, /*keepdim=*/true));
        
        // 拼接池化结果
        auto y = torch::cat({avg_pool, max_pool}, /*dim=*/1);
        
        // 双路卷积并融合
        auto y_conv1 = conv1(y);
        auto y_conv2 = conv2(y);
        auto y_combined = (y_conv1 + y_conv2) / 2;
        
        // 激活并生成注意力掩码
        auto y_sigmoid = sigmoid(y_combined);
        
        // 应用注意力机制
        return x * y_sigmoid;  // 自动广播到输入通道维度
    }
    
    BasicLayerImpl::BasicLayerImpl(int in_channels, 
                    int out_channels, 
                    int kernel_size,
                    int stride,
                    int padding)
    {
        layer = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .padding(padding)
                .stride(stride)
                .dilation(1)
                .bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
        );
        register_module("layer", layer);
    }

    torch::Tensor BasicLayerImpl::forward(torch::Tensor x) 
    {
        return layer->forward(x);
    }

    XFeatModel::XFeatModel()
    {
        norm = torch::nn::InstanceNorm2d(1);

        // CNN Backbone and Heads

        skip1 = torch::nn::Sequential(
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 24, 1).stride(1).padding(0))
        );
        
        block1 = torch::nn::Sequential(
            BasicLayer(1,  4, 3, 1, 1),
            BasicLayer(4,  8, 3, 2, 1),
            BasicLayer(8,  8, 3, 1, 1),
            BasicLayer(8, 24, 3, 2, 1)
        );

        block2 = torch::nn::Sequential(
            BasicLayer(24, 24, 3, 1, 1),
            BasicLayer(24, 24, 3, 1, 1)
        );

        block3 = torch::nn::Sequential(
            BasicLayer(24, 64, 3, 2, 1),
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 1, 1, 0)
        );

        block6=torch::nn::Sequential(
            torch::nn::MaxPool2d(2)
        );
        block7=torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).groups(128)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true)
        );
        block8=torch::nn::Sequential(
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).groups(128)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true)
        );
        block9=torch::nn::Sequential(  torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 1).stride(1).padding(0)));
        block10=torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).groups(128)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(128).affine(false)),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 1).stride(1).padding(0))
        );

        block11=torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(64).affine(false)),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).groups(64)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(64).affine(false)),
            torch::nn::ReLU(true)
        );

        block_fusion = torch::nn::Sequential(
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 3, 1, 1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).padding(0))
        );

        heatmap_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1)),
            torch::nn::Sigmoid()
        );

        keypoint_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 65, 1))
        );

        // Fine Matcher MLP
        mba_block=MBABlock();

        fine_matcher = torch::nn::Sequential(
            torch::nn::Linear(128, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 128)
        );


        register_module("norm", norm);
        register_module("skip1", skip1);
        register_module("block1", block1);
        register_module("block2", block2);
        register_module("block3", block3);
        register_module("block6", block6);
        register_module("block7", block7);
        register_module("block8", block8);
        register_module("block9", block9);
        register_module("block10", block10);
        register_module("block11", block11);
        register_module("block_fusion", block_fusion);
        register_module("heatmap_head", heatmap_head);
        register_module("keypoint_head", keypoint_head);
        register_module("fine_matcher", fine_matcher);
        register_module("mba_block",mba_block);
    }

    torch::Tensor XFeatModel::unfold2d(torch::Tensor x, int ws)
    {   
        auto shape = x.sizes();
        int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
        // 计算新的高度和宽度，确保它们是整数
        int H_new = H / ws;
        int W_new = W / ws;
        int ws_sq = ws * ws;
        // 使用 unfold 操作并 reshape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws);  // 在第2维和第3维进行unfold
        x = x.reshape({B, C, H_new, W_new, ws_sq});  // 重塑为 (B, C, H_new, W_new, ws^2)
        // 调整维度顺序并再次 reshape
        x = x.permute({0, 1, 4, 2, 3}).reshape({B, -1, H_new, W_new});  // 扁平化channels并重新reshape
        return x;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> XFeatModel::forward(torch::Tensor x) 
    {   
        /* 
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats      -> torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints  -> torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap    -> torch.Tensor(B,  1, H/8, W/8) reliability map
        */

        // don't backprop through normalization
        torch::NoGradGuard no_grad;
        x = x.mean(1, true);
        x = norm->forward(x);

        // main backbone
        torch::Tensor x1 = block1->forward(x);
        //std::cout << "x1 shape: " << x1.sizes() << std::endl;
        torch::Tensor x2 = block2->forward(x1 + skip1->forward(x));
        //std::cout << "x2 shape: " << x2.sizes() << std::endl;
        torch::Tensor x3 = block3->forward(x2);
        //std::cout << "x3 shape: " << x3.sizes() << std::endl;
        torch::Tensor x4 = block7->forward(block6->forward(x3));  
        //std::cout << "x4 shape: " << x4.sizes() << std::endl;
        torch::Tensor x5 = block8->forward(x4); 
        //std::cout << "x5 shape: " << x5.sizes() << std::endl;
        std::vector<int64_t> size_array = {x4.size(2), x4.size(3)};
        torch::Tensor x6 = torch::nn::functional::interpolate(block9->forward(x5), 
                                                            torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        torch::Tensor x7 = torch::cat({x6, x4}, 1);  // dim=1 表示在通道维度拼接
        //std::cout << "x7 shape: " << x7.sizes() << std::endl;
        std::vector<int64_t> size_array2 = {x3.size(2), x3.size(3)};
        torch::Tensor x8 = torch::nn::functional::interpolate(block10->forward(x7), 
                                                            torch::nn::functional::InterpolateFuncOptions().size(size_array2)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        torch::Tensor x9 = torch::cat({x8, x3}, 1);  // dim=1 表示在通道维度拼接
        //std::cout << "x9 shape: " << x9.sizes() << std::endl;
        torch::Tensor x10 = block11->forward(x9);  // 通过 block11
        //std::cout << "x10 shape: " << x10.sizes() << std::endl;
        torch::Tensor feats = block_fusion->forward(x10);  // 通过 block_fusion 获取特征
        //std::cout << "feats shape: " << feats.sizes() << std::endl;
        // 通过 heads 获取 heatmap 和 keypoints
        torch::Tensor heatmap = heatmap_head->forward(feats);  // 通过 heatmap_head 得到 Reliability map
        //std::cout << "unfold shape: " << unfold2d(x).sizes() << std::endl;
        torch::Tensor keypoints = keypoint_head->forward(mba_block->forward(unfold2d(x)) + unfold2d(x));  // 计算 keypoints
        //std::cout << "sucess:"<<std::endl;
        return std::make_tuple(feats, keypoints, heatmap);

    }

    //////////////////////////////// InterpolateSparse2d /////////////////////////////////
    InterpolateSparse2d::InterpolateSparse2d(const std::string& mode, bool align_corners)
    : mode(mode), align_corners(align_corners)
    {
    }

    torch::Tensor InterpolateSparse2d::normgrid(torch::Tensor x, int H, int W)
    {
        // normalize coordinates to [-1, 1]
        torch::Tensor size_tensor = torch::tensor({W - 1, H - 1}, x.options());
        return 2.0 * (x / size_tensor) - 1.0;
    }

    torch::Tensor InterpolateSparse2d::forward(torch::Tensor x, torch::Tensor pos, int H, int W)
    {
        // normalize the positions
        torch::Tensor grid = normgrid(pos, H, W).unsqueeze(-2).to(x.dtype());

        // grid sampling  ---- EDIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (mode == "bilinear")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(align_corners));
        }
        else if (mode == "nearest")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kNearest).align_corners(align_corners));
        }   
        else
        {
            std::cerr << "Choose either 'bilinear' or 'nearest'." << std::endl;
            exit(EXIT_FAILURE);
        }

        //reshape output to [B, N, C]
        return x.permute({0, 2, 3, 1}).squeeze(-2);
    }

} // namespace ORB_SLAM3 