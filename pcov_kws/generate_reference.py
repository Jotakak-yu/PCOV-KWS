"""
Can be run directly in cli 
python -m pcov_kws.generate_reference --input-dir ./wakewords --output-dir ./pcov_kws/sample_refs --model-type tcarc14
"""
import os , glob
import numpy as np
import json
from pcov_kws.package_installation_scripts import check_install_librosa
from pcov_kws.audio_processing import (
    ModelType,
    MODEL_TYPE_MAPPER
)

check_install_librosa()

import librosa
import typer
from rich.progress import track

def normalize_audio_rms(audio_data, target_db=-20, max_gain_db=30.0):
    """
    对音频数据进行RMS归一化处理，将有效值调整到目标分贝水平
    
    Args:
        audio_data: 音频波形数据
        target_db: 目标RMS分贝值 (负值，通常-20dB是语音的合理值)
        max_gain_db: 允许的最大增益，防止过度放大安静信号
        
    Returns:
        归一化后的音频数据
    """
    # 防止处理空信号
    if len(audio_data) == 0 or np.all(audio_data == 0):
        return audio_data
        
    # 计算当前RMS值
    rms = np.sqrt(np.mean(audio_data**2))
    
    # 转换为分贝
    if rms > 0:
        rms_db = 20 * np.log10(rms)
    else:
        return audio_data  # 避免对全零信号进行处理
    
    # 计算需要的增益
    gain_db = target_db - rms_db
    
    # 限制最大增益
    gain_db = min(max_gain_db, gain_db)
    
    # 应用增益
    return audio_data * (10 ** (gain_db / 20))

def generate_reference_file_multiple_wakewords(
        input_dir:str = typer.Option(...),
        output_dir:str = typer.Option(...),
        model_type:ModelType = typer.Option(..., case_sensitive=False),
        target_db:float = typer.Option(-20, help="目标RMS分贝值，负值，通常-20dB适合语音"),
        max_gain_db:float = typer.Option(30.0, help="允许的最大增益，防止过度放大安静信号"),
        debug:bool=typer.Option(False)
    ):

    model = MODEL_TYPE_MAPPER[model_type.value]()

    assert(os.path.isdir(input_dir))
    assert(os.path.isdir(output_dir))

    # Creates a new directory for each model type in the output directory
    model_output_dir = os.path.join(output_dir, model_type.value)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
   
    wakeword_dirs = os.listdir(input_dir)

    for wakeword in wakeword_dirs:
        wakeword_input_dir = os.path.join(input_dir, wakeword)
        embeddings = []

        audio_files = [
            *glob.glob(wakeword_input_dir+"/*.mp3"),
            *glob.glob(wakeword_input_dir+"/*.wav")
        ]

        for audio_file in track(audio_files, description=f"Generating Embeddings for {wakeword}.. "):
            x,_ = librosa.load(audio_file,sr=16000)
            
            # 使用RMS归一化处理音频
            original_rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10)
            x = normalize_audio_rms(x, target_db=target_db, max_gain_db=max_gain_db)
            normalized_rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10)
            
            if debug:
                print(f"Audio file: {audio_file}")
                print(f"  Shape: {x.shape}, Range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"  RMS before: {original_rms_db:.2f} dB, after: {normalized_rms_db:.2f} dB")
            
            embeddings.append(
                    model.audioToVector(
                        model.fixPaddingIssues(x)
                    )
                )
            # 只打印第一个embedding的形状
            if len(embeddings) == 1:
                print(f"First embedding shape: {embeddings[0].shape}")

        embeddings = np.squeeze(np.array(embeddings))
        print(f"All embeddings shape: {embeddings.shape}")

        if(debug):
            distanceMatrix = []

            for embedding in embeddings :
                distanceMatrix.append(
                    np.sqrt(np.sum((embedding-embeddings)**2,axis=1))
                )

            temp = np.squeeze(distanceMatrix).astype(np.float16)
            temp2 = temp.flatten()
            print(f"Distance stats - STD: {np.std(temp2):.4f}, Mean: {np.mean(temp2):.4f}")
            print(f"Distance matrix (sample):\n{temp[:5,:5]}")

        # Update output directory to include model type
        open(os.path.join(model_output_dir, f"{wakeword}.json") ,'w').write(
                json.dumps(
                    {
                        "embeddings":embeddings.astype(float).tolist(),
                        "model_type":model_type.value
                        }
                    )
                )
        print(f"Reference file generated: {os.path.join(model_output_dir, wakeword)}.json")

if __name__ == "__main__":
    typer.run(generate_reference_file_multiple_wakewords)