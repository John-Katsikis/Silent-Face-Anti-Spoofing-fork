import subprocess

import FAKEorREAL

def runModel(image, modelDirectory, device ):
   cmd = [
            "python", "FAKEorREAL.py",
            "--image_name", image,
            "--model_dir", modelDirectory,
            "--device_id", str(device)
        ]
   result = subprocess.run(cmd, capture_output=True, text=True)
   return result.stdout

if __name__ == "__main__":
        output = runModel("image_F1.jpg", "./resources/anti_spoof_models", 0)
        print("face is ")
