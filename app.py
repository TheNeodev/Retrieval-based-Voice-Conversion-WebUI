import gradio as gr
from original import *
from audio_separator.separator import Separator
import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

pretraineds_custom_path = os.path.join(
    now_dir, "assets", "pretrained_v2"
)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]


pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")


def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )


def separate_audio(input_file, selected_model):
    # Initialize the Separator class
    separator = Separator()

    # Load the selected model
    separator.load_model(selected_model)

    # Perform separation
    output_files = separator.separate(input_file, 'stem1', 'stem2')
    return f"Separation complete! Output file(s): {' '.join(output_files)}"

# Define model options
model_options = [
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt", 
    "deverb_bs_roformer_8_384dim_10depth", 
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", 
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", 
    "Mel-Roformer-Crowd-Aufr33-Viperx", 
    "Mel-Roformer-Denoise-Aufr33", 
    "Mel-Roformer-Denoise-Aufr33-Aggr", 
    "MDX23C_D1581.ckpt", 
    "MDX23C-8KFFT-InstVoc_HQ.ckpt", 
    "UVR-MDX-NET_Main_340.onnx", 
    "UVR-MDX-NET_Main_390.onnx",
    # Add the remaining model options here
]




with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("# RVC WebUI")
    gr.Markdown(
        value=i18n(
            "This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any codes and files within the software package. See the root directory <b>Agreement-LICENSE.txt</b> for details."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("Model Inference")):
            with gr.Row():
                sid0 = gr.Dropdown(
                    label=i18n("Inferencing voice"), choices=sorted(names)
                )
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("Refresh voice list and index path"), variant="primary"
                    )
                    clean_button = gr.Button(
                        i18n("Unload model to save GPU memory"), variant="primary"
                    )
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Select Speaker/Singer ID"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            modelinfo = gr.Textbox(label=i18n("Model info"), max_lines=8)
            with gr.TabItem(i18n("Single inference")):
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            value=0,
                        )
                        input_audio0 = gr.Audio(
                            label=i18n("The audio file to be processed"),
                            type="filepath",
                        )
                        file_index2 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        file_index1 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                        )
                    with gr.Column():
                        f0method0 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                            ),
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                            ),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio"),
                            value=0.75,
                            interactive=True,
                        )
                        f0_file = gr.File(
                            label=i18n(
                                "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation"
                            ),
                            visible=False,
                        )
                        but0 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Export audio (click on the three dots in the lower right corner to download)"
                            )
                        )

                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index2],
                            api_name="infer_refresh",
                        )

                vc_output1 = gr.Textbox(label=i18n("Output information"))

                but0.click(
                    vc.vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
            with gr.TabItem(i18n("Batch inference")):
                gr.Markdown(
                    value=i18n(
                        "Batch conversion. Enter the folder containing the audio files to be converted or upload multiple audio files. The converted audio will be output in the specified folder (default: 'opt')."
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            value=0,
                        )
                        dir_input = gr.Textbox(
                            label=i18n(
                                "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager)"
                            ),
                            placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                        )
                        inputs = gr.File(
                            file_count="multiple",
                            label=i18n(
                                "Multiple audio files can also be imported. If a folder path exists, this input is ignored."
                            ),
                        )
                        opt_input = gr.Textbox(
                            label=i18n("Specify output folder"), value="opt"
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        file_index3 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )

                    with gr.Column():
                        f0method1 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                            ),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio"),
                            value=1,
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("Export file format"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("Output information"))

                but1.click(
                    vc.vc_multi,
                    [
                        spk_item,
                        dir_input,
                        opt_input,
                        inputs,
                        vc_transform1,
                        f0method1,
                        file_index3,
                        file_index4,
                        # file_big_npy2,
                        index_rate2,
                        filter_radius1,
                        resample_sr1,
                        rms_mix_rate1,
                        protect1,
                        format1,
                    ],
                    [vc_output3],
                    api_name="infer_convert_batch",
                )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1, file_index2, file_index4],
                    outputs=[
                        spk_item,
                        protect0,
                        protect1,
                        file_index2,
                        file_index4,
                        modelinfo,
                    ],
                    api_name="infer_change_voice",
                )
        with gr.TabItem(
            i18n("Vocals/Accompaniment Separation & Reverberation Removal")
        ):
            
            with gr.Row():
                with gr.Column():
                    input_file = gr.File(label="Upload Audio File")
                    model_dropdown = gr.Dropdown(choices=model_options, label="Select Model")
    
                with gr.Column():
                    output_message = gr.Textbox(label="Output Message", interactive=False)
                
                with gr.Row():
                    separate_button = gr.Button("Separate Audio")
                separate_button.click(
                    separate_audio, 
                    inputs=[input_file, model_dropdown], 
                    outputs=[output_message]
                )
        with gr.TabItem(i18n("Train")):
            gr.Markdown(
                value=i18n(
                    "### Step 1. Fill in the experimental configuration.\nExperimental data is stored in the 'logs' folder, with each experiment having a separate folder. Manually enter the experiment name path, which contains the experimental configuration, logs, and trained model files."
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(
                    label=i18n("Enter the experiment name"), value="mi-test"
                )
                author = gr.Textbox(label=i18n("Model Author (Nullable)"))
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n(
                        "Number of CPU processes used for pitch extraction and data processing"
                    ),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Row():
                sr2 = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["32k", "40k", "48k"],
                    value="48k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n(
                        "Whether the model has pitch guidance (required for singing, optional for speech)"
                    ),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("Yes"),
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
            gr.Markdown(
                value=i18n(
                    "### Step 2. Audio processing. \n#### 1. Slicing.\nAutomatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported."
                )
            )
            with gr.Row():
                with gr.Column():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Enter the path of the training folder"),
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("Please specify the speaker/singer ID"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("Process data"), variant="primary")
                with gr.Column():
                    info1 = gr.Textbox(label=i18n("Output information"), value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            gr.Markdown(
                value=i18n(
                    "#### 2. Feature extraction.\nUse CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index)."
                )
            )
            with gr.Row():
                with gr.Column():
                    gpu_info9 = gr.Textbox(
                        label=i18n("GPU Information"),
                        value=gpu_info,
                    )
                    f0method8 = gr.Radio(
                        label=i18n(
                            "Select the pitch extraction algorithm: when extracting singing, you can use 'pm' to speed up. For high-quality speech with fast performance, but worse CPU usage, you can use 'dio'. 'harvest' results in better quality but is slower.  'rmvpe' has the best results and consumes less CPU/GPU"
                        ),
                        choices=["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"],
                        value="rmvpe",
                        interactive=True,
                    )
                with gr.Column():
                    but2 = gr.Button(i18n("Feature extraction"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Output information"), value="")
                but2.click(
                    extract_f0_feature,
                    [
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                    ],
                    [info2],
                    api_name="train_extract_f0_feature",
                )
            gr.Markdown(
                value=i18n(
                    "### Step 3. Start training.\nFill in the training settings and start training the model and index."
                )
            )
            with gr.Row():
                with gr.Column():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("Save frequency (save_every_epoch)"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total training epochs (total_epoch)"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("Batch size per GPU"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n(
                            "Save only the latest '.ckpt' file to save disk space"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "Save a small final model to the 'weights' folder at each save point"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                with gr.Column():
                    pretrained_G14 = gr.Dropdown( 
                        label=("Custom Pretrained G"),
                        info=(
                            "Select the custom pretrained model for the generator."             
                        ),
                        choices=sorted(pretraineds_list_g),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    pretrained_D15 = gr.Dropdown(
                        label=("Custom Pretrained D"),
                        info=("Select the custom pretrained model for the discriminator."),
                        choices=sorted(pretraineds_list_d),     
                        interactive=True,             
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        refresh_custom_pretaineds_button = gr.Button("Refresh Custom Pretraineds")
                        


                    refresh_custom_pretaineds_button.click(
                        fn=refresh_custom_pretraineds,
                        inputs=[],
                        outputs=[pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )

                    but3 = gr.Button(i18n("Train model"), variant="primary")
                    but4 = gr.Button(i18n("Train feature index"), variant="primary")
                    but5 = gr.Button(i18n("One-click training"), variant="primary")
            with gr.Row():
                info3 = gr.Textbox(label=i18n("Output information"), value="")
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start",
                )
                but4.click(train_index, [exp_dir1, version19], info3)
                but5.click(
                    train1key,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        trainset_dir4,
                        spk_id5,
                        np7,
                        f0method8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start_all",
                )

        with gr.TabItem(i18n("ckpt Processing")):
            gr.Markdown(
                value=i18n(
                    "### Model comparison\n> You can get model ID (long) from `View model information` below.\n\nCalculate a similarity between two models."
                )
            )
            with gr.Row():
                with gr.Column():
                    id_a = gr.Textbox(label=i18n("ID of model A (long)"), value="")
                    id_b = gr.Textbox(label=i18n("ID of model B (long)"), value="")
                with gr.Column():
                    butmodelcmp = gr.Button(i18n("Calculate"), variant="primary")
                    infomodelcmp = gr.Textbox(
                        label=i18n("Similarity (from 0 to 1)"),
                        value="",
                        max_lines=1,
                    )
            butmodelcmp.click(
                hash_similarity,
                [
                    id_a,
                    id_b,
                ],
                infomodelcmp,
                api_name="ckpt_merge",
            )

            gr.Markdown(
                value=i18n("### Model fusion\nCan be used to test timbre fusion.")
            )
            with gr.Row():
                with gr.Column():
                    ckpt_a = gr.Textbox(
                        label=i18n("Path to Model A"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("Path to Model B"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Weight (w) for Model A"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Column():
                    sr_ = gr.Radio(
                        label=i18n("Target sample rate"),
                        choices=["32k", "40k", "48k"],
                        value="48k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("Whether the model has pitch guidance"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("Yes"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Column():
                    name_to_save0 = gr.Textbox(
                        label=i18n("Saved model name (without extension)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("Model architecture version"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                    but6 = gr.Button(i18n("Fusion"), variant="primary")
            with gr.Row():
                info4 = gr.Textbox(label=i18n("Output information"), value="")
            but6.click(
                merge,
                [
                    ckpt_a,
                    ckpt_b,
                    alpha_a,
                    sr_,
                    if_f0_,
                    info__,
                    name_to_save0,
                    version_2,
                ],
                info4,
                api_name="ckpt_merge",
            )  # def merge(path1,path2,alpha1,sr,f0,info):

            gr.Markdown(
                value=i18n(
                    "### Modify model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("Path to Model"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("Model information to be modified"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("Save file name (default: same as the source file)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but7 = gr.Button(i18n("Modify"), variant="primary")
                    info5 = gr.Textbox(label=i18n("Output information"), value="")
            but7.click(
                change_info,
                [ckpt_path0, info_, name_to_save1],
                info5,
                api_name="ckpt_modify",
            )

            gr.Markdown(
                value=i18n(
                    "### View model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path1 = gr.File(label=i18n("Path to Model"))
                    but8 = gr.Button(i18n("View"), variant="primary")
                with gr.Column():
                    info6 = gr.Textbox(label=i18n("Output information"), value="")
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")

            gr.Markdown(
                value=i18n(
                    "### Model extraction\n> Enter the path of the large file model under the 'logs' folder.\n\nThis is useful if you want to stop training halfway and manually extract and save a small model file, or if you want to test an intermediate model."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("Path to Model"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("Save name"), value="", interactive=True
                    )
                    with gr.Row():
                        sr__ = gr.Radio(
                            label=i18n("Target sample rate"),
                            choices=["32k", "40k", "48k"],
                            value="48k",
                            interactive=True,
                        )
                        if_f0__ = gr.Radio(
                            label=i18n(
                                "Whether the model has pitch guidance (1: yes, 0: no)"
                            ),
                            choices=["1", "0"],
                            value="1",
                            interactive=True,
                        )
                        version_1 = gr.Radio(
                            label=i18n("Model architecture version"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                    info___ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    extauthor = gr.Textbox(
                        label=i18n("Model Author"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but9 = gr.Button(i18n("Extract"), variant="primary")
                    info7 = gr.Textbox(label=i18n("Output information"), value="")
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
            but9.click(
                extract_small_model,
                [
                    ckpt_path2,
                    save_name,
                    extauthor,
                    sr__,
                    if_f0__,
                    info___,
                    version_1,
                ],
                info7,
                api_name="ckpt_extract",
            )

        with gr.TabItem(i18n("Export Onnx")):
            with gr.Row():
                ckpt_dir = gr.Textbox(
                    label=i18n("RVC Model Path"), value="", interactive=True
                )
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=i18n("Onnx Export Path"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(i18n("Export Onnx Model"), variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = i18n("FAQ (Frequently Asked Questions)")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "FAQ (Frequently Asked Questions)":
                    with open("docs/cn/faq.md", "r", encoding="utf8") as f:
                        info = f.read()
                else:
                    with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                        info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

try:
    import signal

    def cleanup(signum, frame):
        signame = signal.Signals(signum).name
        print(f"Got signal {signame} ({signum})")
        app.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    if config.global_link:
        app.queue(max_size=1022).launch(share=True, max_threads=511)
    else:
        app.queue(max_size=1022).launch(
            max_threads=511,
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
except Exception as e:
    logger.error(str(e))
