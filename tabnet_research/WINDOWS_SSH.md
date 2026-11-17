# Windows: подключение вычислительного хоста

Подключение уже настроено: SSH alias `avito-windows`, Windows Bulat, RTX3060Ti.
Ниже сохранена инструкция первоначальной настройки от 17 ноября 2026.
Актуальные задачи, окружения и продолжение: [WINDOWS_TRAINING_CONTEXT.md](../docs/WINDOWS_TRAINING_CONTEXT.md).

## Первый вход

На Mac создан отдельный ключ `~/.ssh/avito_windows_ed25519`; закрытая часть
остаётся на Mac. Подготовленный локальный файл
`artifacts/windows_ssh_setup.ps1` содержит только открытый ключ.
Он не входит в Git. Перенести этот файл на Windows и выполнить из PowerShell,
запущенной от администратора **под той учётной записью, через которую будем
подключаться**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\windows_ssh_setup.ps1
```

Скрипт устанавливает встроенный OpenSSH Server при необходимости, включает
службу и штатное правило Windows Firewall TCP/22, добавляет открытый ключ
без удаления существующих ключей, задаёт ACL файла администраторских ключей.
Он печатает имя пользователя, IPv4-адреса, отпечаток SSH-хоста и сведения GPU.
Python, драйверы, данные и задания обучения скрипт не изменяет.
Параметры установки: [Microsoft OpenSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse).
Размещение ключей и ACL: [Microsoft key management](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement).

Если SSH уже работает, достаточно добавить открытый ключ: для обычной учётной
записи — `%USERPROFILE%\.ssh\authorized_keys`, для члена Administrators при
штатном `sshd_config` — `%PROGRAMDATA%\ssh\administrators_authorized_keys`.
Кастомный `sshd_config` сначала прочитать; не заменять его целиком.

Нужны адрес, порт, имя пользователя и вывод отпечатка SSH-хоста. Для одной LAN
подходит доступный с Mac локальный IPv4. Для разных сетей нужен уже настроенный
маршрут/VPN; одного адреса Windows из домашней сети недостаточно.

С Mac после проверки отпечатка:

```bash
ssh -i ~/.ssh/avito_windows_ed25519 -o IdentitiesOnly=yes -p PORT USER@HOST
```

Windows PIN не является SSH-паролем. При входе по ключу пароль не передаётся.

## После подключения

Сначала прочитать `nvidia-smi`, список Python-процессов и заданий планировщика,
объём RAM и свободное место SSD. Проверить версию Python и выполнить в выбранном
окружении:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Для чистого нового окружения выбрать Python 3.11; сборку PyTorch CUDA подобрать
после чтения версии драйвера, используя [официальную установку PyTorch](https://pytorch.org/get-started/locally/).
Окружение другой модели не менять. `pytorch-tabnet==4.1.0` закреплён для этого
опыта; сначала проверить зависимости и синтетические тесты.

Код переносить в отдельный каталог TabNet. Готовый research-cache MLP можно
переиспользовать только после проверки metadata/массивов/когорты. В коде есть
замороженный снимок MLP `d4f9edb`; полный prepared train и research-cache —
разные артефакты. Кэш этой пары содержит train4 и selection, а не late/test.
Если готового кэша на новом ПК нет, сначала отдельно подготовить его из train
и val. Штатный runner сам его не создаёт.

Оценка размеров по схеме кэша: около 6,3 GiB для массивов train4+selection
плюс метаданные/накладные расходы. Это расчёт по размерам массивов, не измерение
целевого диска; дополнительно нужны исходники, исходные данные, окружение и
checkpoints. Memmap хранится на SSD, в GPU передаётся текущий батч.

Длинный запуск выполняется отдельной задачей Windows Task Scheduler с уникальным
именем, чтобы разрыв SSH не прервал обучение. Не использовать имена задач MLP,
не останавливать чужие процессы и не запускать конкурирующие GPU-очереди.
Перед модельным опытом нужен короткий CUDA-прогон с конечным loss, градиентами,
шагом optimizer и измерением VRAM/скорости. Синтетические тесты Mac не заменяют
эту проверку. Текущий запрос не запускает серию обучения автоматически.

Карточка пары и команда trainer — [README.md](README.md), журнал —
[tabnet_exp.md](../tabnet_exp.md). Test/submission требуют отдельной команды;
любая публикация GitHub — показа diff и явного одобрения владельца.
