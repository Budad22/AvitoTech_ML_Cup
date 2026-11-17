> LIVE SNAPSHOT: 18 Nov01:11:13MSK production PID22540, CUDA optimizer steps confirmed, epoch1/5,492380total steps. Full profile26.936batch/sec, estimated5.078h; no final metric yet.

# Вычислительный ПК: состояние и продолжение работы

Снимок 18 ноября 2026, МСК. Рабочая ветка `bulat/tabnet-baseline`, отдельный
worktree `/Users/bulat/Documents/Avito/AvitoTech_ML_Cup_tabnet`.
Главный исходный worktree остаётся на `bulat/catboost-baseline`.
Локальные изменения других worktree не переносить и не сбрасывать.

## Что разрабатываем

Начали с TabNet на входах сильной MLP, получили измеренный отрицательный
результат и перешли к воспроизводимому официальному Transformer. Владелец
явно разрешил полный ночной запуск: **512 train-шардов, 5 эпох с нуля**.
Половинный запуск отменён до старта. Автоматический перебор вариантов не задан.

| Опыт | Train | Результат selection NDCG | Решение |
| --- | --- | --- | --- |
| EXP-TABNET-001-r01, matched MLP | 4 шарда | 0.3613337358564908 | контроль |
| EXP-TABNET-001-r01, TabNet | 4 шарда | 0.3344101750341997 | продолжить ограниченно |
| EXP-TABNET-006-r01, TabNet после 32000 шагов | 4 шарда | 0.34839623210726983 | уступает MLP, тюнинг отложен |
| EXP-TRF-001-r01, официальный Transformer | 4 шарда | 0.329371524620737 | малый контроль завершён |
| EXP-TRF-002-r01, официальный Transformer | 512 шардов | ещё нет | полный ночной запуск |

Отчёты предыдущих запусков: `tabnet_research/results/`,
`transformer_research/results/`. Это локальная val; оценку сайта им не приписывать.
Журналы: [tabnet_exp.md](../tabnet_exp.md), [transformer_exp.md](../transformer_exp.md).

## Доступ и задачи

- SSH с Mac: `ssh avito-windows`; LAN-хост `192.168.1.30`, пользователь `bulat`.
- Выделенный ключ `~/.ssh/avito_windows_ed25519` хранится только на Mac.
  Содержимое ключа, пароли и другие секреты в репозитории отсутствуют.
- ПК: Windows, 32 ГБ RAM, RTX3060Ti 8 ГБ, драйвер610.88. OpenSSH слушает22.
- Задача `AvitoTransformer002`, workspace `C:\AvitoBaselines\EXP-TRF-002-r01`.
  S4U/Highest; запускает PowerShell wrapper с Start-Process -Wait и отдельными
  stdout/stderr. Разрыв SSH и отключение ноутбука не останавливают pipeline.
- Старые задачи `AvitoTabNet001`, `AvitoTabNet006`, `AvitoTransformer001`
  завершены; их веса и окружения не менять.
- Питание и интернет ПК нужны; worker предотвращает автоматический сон на время
  работы. Доступ из другой сети требует отдельного маршрута/VPN; LAN-адрес
  не означает настроенный удалённый доступ из интернета.

## Полный ночной Transformer

`full_windows.ps1`: проверить raw → тесты → compact cache → аудит/профиль → train.
Все512 train-шардов скачаны и проверены. Train: **3151208 лент, 246745836 кандидатов**.
Mac передал 12034325865 байт (int8 train/val item-матрицы,64train и64val Parquet)
за147.793с; SHA всех файлов совпали. Остальные448 шардов Windows скачал сам
из официального ZIP с ETag/CRC/SHA. Ноутбук больше не является источником данных.

Сбой первоначального compact build и исправление:

1. `np.prod` на Windows переполнил int32 при проверке10.44GB матрицы.
   Исправлено `dtype=np.int64`; реальный preflight после этого прошёл.
2. В полном train есть повторные кандидаты внутри одной ленты. Например,
   `feed_id=1233995`, `item_id=9455009`, позиции128/129 (с нуля), метки0/0,
   одинаковый embedding_idx70119948. Официальный CandidateData сохраняет их.
   Избыточная наша проверка остановила подготовку в01:04:39МСК.
   Исправлено: train повторяется в точности как в исходнике, без дедупликации;
   число повторов пишется в metadata.diagnostics. На всём train сохранено4 повторные строки; вval повторов0. Для val уникальность обязательна.
   Тест проверяет сохранение повторов/меток вtrain и запрет повторов вval.
   Все6 тестов на Windows прошли за2.080с, задача перезапущена около01:07МСК.
   Основная модель до этого сбоя не обучалась.

Полный raw: `C:\AvitoBaselines\datasets\full-v1\raw`;
compact: `C:\AvitoBaselines\datasets\full-v1\compact`.
Compact хранит f1..f5, category, target, item_index/item_id, offsets/feed/time/user
и один int8 user128 на ленту; item int8-матрица общая, читается memmap.
Нормализация user/item — исходная F.normalize float32 в Dataset, без повторения
519-мерного float-вектора на диске. Колонки target/ID/pos не входят в признаки.

Словарь категорий строится только на train. Selection прежняя:
`timestamp < 1790249984`,11534 ленты/1101295 кандидатов,fast2000/confirmation9534.
Вся val исторически использована; независимым holdout её не называть.
Перед тренировкой проверяются ключи/offsets/метки и все selection-входы против
старого общего кэша. На 4-шардовом preflight max_input_error=0.0,40CUDA-батчей
за1.187с (33.6984batch/s), GPUpeak110569472bytes; веса выброшены.
Оценка полного492380шагового обучения4.059ч по короткому замеру, плюс подготовка;
пользователю дан запас5–7ч. Это прогноз, не измеренная длительность полного run.

Обучение: исходные Ranker/collate без изменений,84841 параметр,width64,layers2,
heads4,dropout0.1,category embedding8; BCE по реальным кандидатам,AdamW
lr0.001/weight_decay0.0001,batch32,seed42,5epochs,безAMP/clipping/history.
Checkpoint выбирается по minimum fast2000 row Logloss. После freeze считается
full NDCG selection/confirmation, score=float32 sigmoid. NDCG по всей ленте,
равный вес лент, пессимистические ties, для нулевых positives —1.

Окружение001 читается без изменения:
`C:\AvitoBaselines\EXP-TRF-001-r01\.venv\Scripts\python.exe`,Python3.11.9,
Torch2.10.0+cu128,NumPy1.26.4,Polars1.14.0,Click8.1.8,Loguru0.7.3,CatBoost1.2.10.
PyArrow21 установлен отдельно в002/dependencies через uv --target --no-deps;
для ручного запуска выставить PYTHONPATH на этот каталог.
Лимит trainer8ч, задачи12ч; достижение бюджета сохраняет last.pt и означает
незавершённый опыт. Checkpoint каждые100шагов и в конце эпохи; resume восстанавливает
optimizer/RNG/nativeRandomSampler без перечитывания пройденных feed.

## Где смотреть и как продолжить

PowerShell на Windows:

```powershell
$root='C:\AvitoBaselines\EXP-TRF-002-r01'
Get-ScheduledTask -TaskName AvitoTransformer002
Get-Content "$root\pipeline-status.json"
Get-Content "$root\worker.log" -Tail 12
Get-Content "$root\worker.err.log" -Tail 12
Get-Content "$root\loader-profile.json"
Get-Content "$root\runs\EXP-TRF-002-r01\status.json"
```

`worker.err.log` содержит также штатный вывод установщика и unittest; смотреть
текущий pipeline stage, traceback и trainer status вместе, а не только имя stderr.
Во время download/build GPU почти свободен. Во время CUDA смотреть график CUDA/
Compute в диспетчере задач; график3D может не показывать вычислительную нагрузку.

Не запускать второй trainer поверх текущего. Если pipeline завершился ошибкой,
сначала установить причину по логам. Для того же кода/конфига и исправленной
внешней причины повторный Start-ScheduledTask продолжает сохранённые данные.
Завершённый compact имеет checksum/identity guard; не удалять metadata наугад.
Если train уже начался, не менять его source/config: checkpoint проверяет SHA.
Пауза по команде владельца — создать STOP_AFTER_CHECKPOINT в output trainer.
Для продолжения убрать STOP-файл и повторить тот же pipeline/команду.

После завершения: получить config,status,report,frozen,fit-result иscores/per-feed
из runs/EXP-TRF-002-r01; независимо пересчитать NDCG, сверить SHA checkpoint,
cohort,критерийвыбора и обновить transformer_exp.md. До результата не обещать0.39.
Большие прогнозы/checkpoint/raw/compact остаются вне Git.

Источники для восстановления:

- Официальный baseline ZIP: https://storage.yandexcloud.net/avitotech-mlcup-2026/baseline.zip
  SHA256 `0bcbed69e4aefb169f05d7dbec080c42c181ec5b587227c64219465b8edde309`.
- neural_baseline.py SHA256
  `87b731df2a5507957c075e93a7f219c1fa4f59b9b7bb9752c59c43087bb2cc06`.
- Train URL/ETag/512 memberCRC и размеры: `transformer_research/configs/full-source.json`.
- SHA/размеры перенесённых файлов: `transformer_research/configs/transfer_manifest.json`.
  Это метаданные, не сами датасеты.
- Локальные исходники official: `artifacts/official-baseline-review/`; на ПК —002/official.
- Общий старый кэш: `C:\AvitoTabNet\EXP-TABNET-001-r01\artifacts\research-cache-train4-v1`.
- Замороженный MLP source commit `d4f9edb`, Mac artifacts/mlp-source-d4f9edb;
  TabNet4.1.0 в отдельном target. Полные команды — README соответствующих модулей.

## Что разрешено

Текущий запрос разрешает обучение полного train и оценку val, сохранение кода,
документации и локальный commit. Test-инференс/submission требуют отдельной команды.
Push/любая публикация GitHub — только после показа конкретного diff и явного
одобрения владельца по AGENTS.md. Запуск train не означает разрешение submission.
