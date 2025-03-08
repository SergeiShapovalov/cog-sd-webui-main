# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os, sys, json
import shutil
import time
import subprocess  # Для запуска внешних процессов

sys.path.extend(["/stable-diffusion-webui-forge-main"])

from cog import BasePredictor, BaseModel, Input, Path
def download_base_weights(url: str, dest: Path):
    """
    Загружает базовые веса модели.
    
    Args:
        url: URL для загрузки весов
        dest: Путь для сохранения весов
    """
    start = time.time()  # Засекаем время начала загрузки
    print("downloading url: ", url)
    print("downloading to: ", dest)
    # Используем pget для эффективной загрузки файлов
    # Убираем параметр -xf, так как файл не является архивом
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # Выводим время загрузки

class Predictor(BasePredictor):
    def _move_model_to_sdwebui_dir(self):
        """
        Проверяет наличие модели и загружает ее, если она отсутствует.
        Модель должна быть предварительно загружена во время сборки в cog.yaml.
        """
        target_dir = "/stable-diffusion-webui-forge-main/models/Stable-diffusion"
        model_path = os.path.join(target_dir, "flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors")
        
        # Проверяем, существует ли уже файл модели
        if os.path.exists(model_path):
            print(f"Модель уже загружена: {model_path}")
            return
        
        # Если модель не найдена, загружаем ее
        print("Модель не найдена, загружаем...")
        os.makedirs(target_dir, exist_ok=True)
        download_base_weights(
            "https://civitai.com/api/download/models/819165?type=Model&format=SafeTensor&size=full&fp=nf4&token=18b51174c4d9ae0451a3dedce1946ce3",
            model_path
        )
    def _download_loras(self, lora_urls):
        """
        Загружает LoRA файлы по указанным URL.
        
        Args:
            lora_urls: Список URL для загрузки LoRA файлов
            
        Returns:
            Список путей к загруженным LoRA файлам
        """
        if not lora_urls or lora_urls.strip() == "":
            return []
            
        lora_urls_list = [url.strip() for url in lora_urls.split(",") if url.strip()]
        if not lora_urls_list:
            return []
            
        target_dir = "/stable-diffusion-webui-forge-main/models/Lora"
        os.makedirs(target_dir, exist_ok=True)
        
        lora_paths = []
        for i, url in enumerate(lora_urls_list):
            try:
                # Извлекаем имя файла из URL или используем индекс, если не удалось
                filename = os.path.basename(url.split("?")[0])
                if not filename or filename == "":
                    filename = f"lora_{i+1}.safetensors"
                
                # Убедимся, что файл имеет расширение .safetensors
                if not filename.endswith(".safetensors"):
                    filename += ".safetensors"
                
                lora_path = os.path.join(target_dir, filename)
                download_base_weights(url, lora_path)
                lora_paths.append(lora_path)
                print(f"LoRA {i+1} успешно загружена: {lora_path}")
            except Exception as e:
                print(f"Ошибка при загрузке LoRA {i+1}: {e}")
        
        return lora_paths

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self._move_model_to_sdwebui_dir()

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
        from modules import timer
        
        # Безопасный импорт memory_management
        try:
            from backend import memory_management
            self.has_memory_management = True
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать memory_management: {e}")
            self.has_memory_management = False
        
        # moved env preparation to build time to reduce the warm-up time
        # from modules import launch_utils

        # with launch_utils.startup_timer.subcategory("prepare environment"):
        #     launch_utils.prepare_environment()

        from modules import initialize_util
        from modules import initialize

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()

        initialize.check_versions()

        initialize.initialize()
        
        # Импортируем shared после initialize.initialize()
        from modules import shared
        
        # Устанавливаем forge_preset на 'flux'
        shared.opts.set('forge_preset', 'flux')
        
        # Устанавливаем unet тип на 'Automatic' для Flux
        shared.opts.set('forge_unet_storage_dtype', 'bnb-nf4')
        
        # Устанавливаем чекпоинт
        shared.opts.set('sd_model_checkpoint', 'flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4.safetensors')
        
        # Оптимизация памяти для лучшего качества и скорости с Flux
        if self.has_memory_management:
            # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
            total_vram = memory_management.total_vram
            inference_memory = int(total_vram * 0.1)  # 10% для вычислений
            model_memory = total_vram - inference_memory
            
            memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Конвертация в байты
            print(f"[GPU Setting] Выделено {model_memory} MB для весов модели и {inference_memory} MB для вычислений")
            
            # Настройка Swap Method на ASYNC для лучшей производительности
            try:
                from backend import stream
                # Для Flux рекомендуется ASYNC метод, который может быть до 30% быстрее
                stream.stream_activated = True  # True = ASYNC, False = Queue
                print("[GPU Setting] Установлен ASYNC метод загрузки для лучшей производительности")
                
                # Настройка Swap Location на Shared для лучшей производительности
                memory_management.PIN_SHARED_MEMORY = True  # True = Shared, False = CPU
                print("[GPU Setting] Установлен Shared метод хранения для лучшей производительности")
            except ImportError as e:
                print(f"Предупреждение: Не удалось импортировать stream: {e}")
        else:
            print("[GPU Setting] memory_management не доступен, используются настройки по умолчанию")

        from fastapi import FastAPI

        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock

        self.api = Api(app, queue_lock)

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(
            description="Negative Prompt (для Flux рекомендуется оставить пустым и использовать Distilled CFG)",
            default="",
        ),
        width: int = Input(
            description="Width of output image", ge=1, le=1024, default=512
        ),
        height: int = Input(
            description="Height of output image", ge=1, le=1024, default=768
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        sampler: str = Input(
            description="Sampling method для Flux моделей",
            choices=[
                "[Forge] Flux Realistic",
                "Euler",
                "Euler a",
                "DPM++ 2M",
                "DPM++ SDE",
                "DPM++ 2M SDE",
                "DPM++ 2M SDE Karras",
                "DPM++ 2M SDE Exponential",
                "DPM++ 3M SDE",
                "DPM++ 3M SDE Karras",
                "DPM++ 3M SDE Exponential"
            ],
            default="[Forge] Flux Realistic",
        ),
        scheduler: str = Input(
            description="Schedule type для Flux моделей",
            choices=[
                "Simple",
                "Karras",
                "Exponential",
                "SGM Uniform",
                "SGM Karras",
                "SGM Exponential",
                "Align Your Steps",
                "Align Your Steps 11",
                "Align Your Steps 32",
                "Align Your Steps GITS",
                "KL Optimal",
                "Normal",
                "DDIM",
                "Beta",
                "Turbo"
            ],
            default="Simple",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=20
        ),
        guidance_scale: float = Input(
            description="CFG Scale (для Flux рекомендуется значение 1.0)", ge=1, le=50, default=1.0
        ),
        distilled_guidance_scale: float = Input(
            description="Distilled CFG Scale (основной параметр для Flux, рекомендуется 3.5)", ge=0, le=30, default=3.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=-1
        ),
        # image: Path = Input(description="Grayscale input image"),
        enable_hr: bool = Input(
            description="Hires. fix",
            default=False,
        ),
        hr_upscaler: str = Input(
            description="Upscaler for Hires. fix",
            choices=[
                "Latent",
                "Latent (antialiased)",
                "Latent (bicubic)",
                "Latent (bicubic antialiased)",
                "Latent (nearest)",
                "Latent (nearest-exact)",
                "None",
                "Lanczos",
                "Nearest",
                "ESRGAN_4x",
                "LDSR",
                "R-ESRGAN 4x+",
                "R-ESRGAN 4x+ Anime6B",
                "ScuNET GAN",
                "ScuNET PSNR",
                "SwinIR 4x",
            ],
            default="Latent",
        ),
        hr_steps: int = Input(
            description="Inference steps for Hires. fix", ge=0, le=100, default=20
        ),
        hr_scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=2
        ),
        denoising_strength: float = Input(
            description="Denoising strength. 1.0 corresponds to full destruction of information in init image",
            ge=0,
            le=1,
            default=0.5,
        ),
        enable_adetailer: bool = Input(
            description="ADetailer (не рекомендуется для Flux моделей)",
            default=False,
        ),
        lora_urls: str = Input(
            description="Ссылки на LoRA файлы, разделенные запятыми (например, https://example.com/lora1.safetensors,https://example.com/lora2.safetensors)",
            default="",
        ),
        lora_weights: str = Input(
            description="Веса для каждой LoRA от 0 до 1, разделенные запятыми (например, 0.7,0.5). Должно соответствовать количеству LoRA",
            default="",
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        # Загружаем LoRA файлы, если они указаны
        lora_files = self._download_loras(lora_urls)
        
        # Парсим веса LoRA
        lora_weights_list = []
        if lora_weights and lora_weights.strip():
            try:
                lora_weights_list = [float(w.strip()) for w in lora_weights.split(",") if w.strip()]
            except ValueError:
                print("Ошибка при парсинге весов LoRA. Используем значения по умолчанию (1.0)")
                lora_weights_list = []
        
        # Если количество весов не соответствует количеству LoRA, используем значение 1.0 для всех
        if len(lora_weights_list) != len(lora_files):
            lora_weights_list = [1.0] * len(lora_files)
        
        # Создаем строку с LoRA для промпта в формате <lora:имя_файла:вес>
        lora_prompt_parts = []
        for i, (lora_path, weight) in enumerate(zip(lora_files, lora_weights_list)):
            lora_name = os.path.basename(lora_path)
            # Удаляем расширение .safetensors из имени файла
            if lora_name.endswith('.safetensors'):
                lora_name = lora_name[:-12]
            lora_prompt_parts.append(f"<lora:{lora_name}:{weight:.2f}>")
            print(f"Добавляем LoRA в промпт: <lora:{lora_name}:{weight:.2f}>")
        
        # Добавляем LoRA в начало промпта
        lora_prompt = " ".join(lora_prompt_parts)
        if lora_prompt:
            prompt = f"{lora_prompt} {prompt}"
        
        payload = {
            # "init_images": [encoded_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": num_outputs,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
            "do_not_save_samples": True,
            "sampler_name": sampler,  # Используем выбранный пользователем sampler
            "scheduler": scheduler,    # Устанавливаем scheduler для Flux
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_steps,
            "denoising_strength": denoising_strength if enable_hr else None,
            "hr_scale": hr_scale,
            "distilled_cfg_scale": distilled_guidance_scale,  # Добавляем параметр distilled_cfg_scale для Flux
            "hr_additional_modules": [],  # Добавляем пустой список для hr_additional_modules, чтобы избежать ошибки
        }
        
        # LoRA уже добавлены в промпт в формате <lora:имя_файла:вес>
        # Нет необходимости добавлять их в payload отдельно

        alwayson_scripts = {}

        if enable_adetailer:
            alwayson_scripts["ADetailer"] = {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                    }
                ],
            }

        # Добавляем все скрипты в payload, если они есть
        if alwayson_scripts:
            payload["alwayson_scripts"] = alwayson_scripts

        from modules.api.models import (
            StableDiffusionTxt2ImgProcessingAPI,
        )
        
        req = StableDiffusionTxt2ImgProcessingAPI(**payload)
        # generate
        resp = self.api.text2imgapi(req)
        info = json.loads(resp.info)

        from PIL import Image
        import uuid
        import base64
        from io import BytesIO

        outputs = []

        for i, image in enumerate(resp.images):
            seed = info["all_seeds"][i]
            gen_bytes = BytesIO(base64.b64decode(image))
            gen_data = Image.open(gen_bytes)
            filename = "{}-{}.png".format(seed, uuid.uuid1())
            gen_data.save(fp=filename, format="PNG")
            output = Path(filename)
            outputs.append(output)

        return outputs
