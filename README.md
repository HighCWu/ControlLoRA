# ControlLoRA: A Lightweight Neural Network To Control Stable Diffusion Spatial Information

EN | [中文](./README_CN.md)

By combining the ideas of [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) and [cloneofsimo/lora](https://github.com/cloneofsimo/lora), we can easily fine-tune stable diffusion to achieve the purpose of controlling its spatial information, with ControlLoRA, a simple and small (~7M parameters, ~25M storage space) network.

ControlNet is large and it's not easy to send to your friends. With the idea of LoRA, we don't even need to transfer the entire stable diffusion model. Use the 25M ControlLoRA to save your time. 

You could use gradio apps in the `apps` directory to try the pretrained models. More dataset types of models and their supporting gradio apps wanted. The `annotator` directory is borrowed from ControlNet.

You could download some pretrained models from [huggingface](https://huggingface.co/HighCWu/ControlLoRA). Note that I only used 100 MPII pictures for the training of the openpose, so the model effect is not good. So I suggest you train your own ControlLoRA.

## Features & News

2023/02/22 - Add new ControlLora, which decomposites the prompt features and the spatial information with smaller size (~5M parameters, ~20M storage space). You could do something like: training on sd v1.5 then inference on anything v3.0 .

## How To Train

Refer to the script in the `tasks` directory. I highly refer to the training code from [diffusers](https://github.com/huggingface/diffusers).

You could add or modify config file in the `configs` directory to custom the ControlLoRA model architecture. To enhance the effect of the model, you could change some blocks to other residual block types of diffusers and you could increase the number of layers of blocks by modify the config files.

## Work In Progress

- [ ] More type tasks mentioned in ControlNet.

- [ ] Experiment of mixing LoRA and ControlLoRA.

    We could inject pretrained LoRA models before the ControlLoRA. See `mix_lora_and_control_lora.py` for more details. 

    ![p](docs/imgs/mix_13.png)
    *portrait of male HighCWu*

## ControlLoRA with Canny Edge

<strong>sd-diffusiondb-canny-model-control-lora, on 100 openpose pictures, 30k training steps</strong>

Stable Diffusion 1.5 + ControlLoRA (using simple Canny edge detection)

    python apps/gradio_canny2image.py

Highly referred to the ControlNet codes.

The Gradio app also allows you to change the Canny edge thresholds. Just try it for more details.


Prompt: "bird"
![p](docs/imgs/p1.png)

Prompt: "cute dog"
![p](docs/imgs/p2.png)

## ControlLoRA with Human Pose

<strong>sd-mpii-pose-model-control-lora, on 100 openpose pictures, 30k training steps</strong>

Stable Diffusion 1.5 + ControlLoRA (using human pose)

    python apps/gradio_pose2image.py

Highly referred to the ControlNet codes.

Apparently, this model deserves a better UI to directly manipulate pose skeleton. However, again, Gradio is somewhat difficult to customize. Right now you need to input an image and then the Openpose will detect the pose for you.

Note that I only used 100 MPII pictures for the training of the openpose, so the model effect is not good. So I suggest you train your own ControlLoRA.

Prompt: "Chief in the kitchen"
![p](docs/imgs/p11.png)

Prompt: "An astronaut on the moon"
![p](docs/imgs/p12.png)

PS: I don't know why my gallery didn't show the full images and I should click an output to show the full result of one of the outputs, like this: ![p](docs/imgs/p12_gallery.png)

# Discuss together

QQ Group: [艾梦的小群](https://jq.qq.com/?_wv=1027&k=yMtGIF1Q)

QQ Channel: [艾梦的AI造梦堂](https://pd.qq.com/s/1qyek3j0e)

Discord: [AI Players - AI Dream Bakery](https://discord.gg/zcJszfPrZs)

# Citation

    @software{wu2023controllora,
        author = {Wu Hecong},
        month = {2},
        title = {{ControlLoRA: A Lightweight Neural Network To Control Stable Diffusion Spatial Information}},
        url = {https://github.com/HighCWu/ControlLoRA},
        version = {1.0.0},
        year = {2023}
    }
