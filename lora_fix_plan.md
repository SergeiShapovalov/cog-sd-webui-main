# План исправления проблемы с LoRA для модели Flux

## Проблема

При попытке использовать LoRA с моделью Flux возникает ошибка:

```
ValueError: 'sd_forge_lora' is not in list
```

Это происходит потому, что в файле `predict.py` на строках 347-351 используется `alwayson_scripts["sd_forge_lora"]`, но такого скрипта нет в системе. Расширение `sd_forge_lora` не регистрирует скрипт с именем "sd_forge_lora" в системе скриптов WebUI.

## Анализ

1. Расширение `sd_forge_lora` регистрирует только API-эндпоинты через `script_callbacks.on_app_started(api_networks)`.
2. В файле `extra_networks_lora.py` есть класс `ExtraNetworkLora`, который наследуется от `extra_networks.ExtraNetwork`. Этот класс отвечает за активацию и деактивацию LoRA-сетей.
3. В методе `activate` вызывается функция `networks.load_networks`, которая загружает LoRA-сети.
4. В файле `network.py` есть класс `NetworkOnDisk`, который представляет LoRA-файл на диске, и класс `Network`, который представляет загруженную LoRA-сеть.
5. В классе `NetworkOnDisk` есть метод `detect_version`, который определяет версию Stable Diffusion, для которой предназначена LoRA. Там есть проверка на `SdVersion.Flux`, что говорит о том, что LoRA может быть предназначена для модели Flux.

## Решение

Проблема в том, что в файле `predict.py` используется `alwayson_scripts["sd_forge_lora"]`, но такого скрипта нет в системе. Нам нужно найти правильное имя скрипта для использования LoRA с моделью Flux.

Мы проанализировали несколько расширений, которые регистрируют скрипты для использования в API:
- `sd_forge_controlnet` регистрирует скрипт с именем "ControlNet"
- `sd_forge_freeu` регистрирует скрипт с именем "FreeU Integrated (SD 1.x, SD 2.x, SDXL)"
- `sd_forge_dynamic_thresholding` регистрирует скрипт с именем "DynamicThresholding (CFG-Fix) Integrated"
- `sd_forge_stylealign` регистрирует скрипт с именем "StyleAlign Integrated"

Но расширение `sd_forge_lora` не регистрирует скрипт с именем "sd_forge_lora" в системе скриптов WebUI. Вместо этого, оно регистрирует только API-эндпоинты через `script_callbacks.on_app_started(api_networks)`.

### Решение 1: Использовать prompt с тегами LoRA

Вместо использования `alwayson_scripts`, можно добавить теги LoRA в prompt. Например:

```python
prompt = prompt + f"<lora:{lora_name}:{weight}>"
```

Это будет работать, потому что класс `ExtraNetworkLora` обрабатывает теги LoRA в prompt и загружает соответствующие LoRA-сети.

### Решение 2: Использовать правильное имя скрипта

Если мы хотим использовать `alwayson_scripts`, нужно найти правильное имя скрипта. Для этого можно использовать API-эндпоинт `/sdapi/v1/scripts`, который возвращает список доступных скриптов.

Но поскольку расширение `sd_forge_lora` не регистрирует скрипт в системе скриптов WebUI, мы не можем использовать этот подход.

### Решение 3: Напрямую вызывать функцию `networks.load_networks`

Можно напрямую вызывать функцию `networks.load_networks` перед вызовом API:

```python
import networks
networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
```

Но это может быть сложно реализовать, так как функция `networks.load_networks` требует доступа к внутренним структурам данных WebUI.

### Решение 4: Создать скрипт для LoRA

Мы можем создать скрипт для LoRA, который будет наследоваться от `scripts.Script` и переопределять методы `title` и `show`. Метод `show` должен возвращать `scripts.AlwaysVisible`, чтобы скрипт всегда был активен и доступен через API.

Например:

```python
class LoraForForge(scripts.Script):
    sorting_priority = 13

    def title(self):
        return "LoRA Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        return []

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # Здесь можно добавить код для загрузки LoRA
        pass
```

Затем мы можем использовать `alwayson_scripts["LoRA Integrated"]` вместо `alwayson_scripts["sd_forge_lora"]`.

## Рекомендуемое решение

Рекомендуется использовать Решение 1, так как оно наиболее простое и надежное. Вот как это можно реализовать:

```python
# Вместо этого:
if lora_args:
    alwayson_scripts["sd_forge_lora"] = {
        "args": lora_args
    }

# Используем это:
for i, (lora_name, weight, _) in enumerate(lora_args):
    prompt = prompt + f"<lora:{lora_name}:{weight}>"
```

Это позволит использовать LoRA с моделью Flux без изменения кода расширения `sd_forge_lora`.

Однако, если пользователь сообщает, что этот подход не работает, можно попробовать Решение 4 и создать скрипт для LoRA.