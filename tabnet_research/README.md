# TabNet: изолированный GPU-опыт

Сравниваются две заново обучаемые модели: MLP + история + PLR + softmax из
origin/MLP d4f9edb и те же входы/PLR/loss с TabNet вместо MLP backbone.
Код друга, работающие задачи и исходные данные не изменяются.

Windows подключён через SSH alias `avito-windows`, CUDA проверена на RTX3060Ti.
EXP-TABNET-001 и 006 завершены. В 006 TabNet selection0.348396 уступил
matched MLP0.361334; дальнейший автоматический тюнинг отложен.
Текущий ночной запуск — Transformer EXP-TRF-002 на полном train,
см. [состояние Windows](../docs/WINDOWS_TRAINING_CONTEXT.md).

Для обеих моделей фиксированы train4, batch16 целых лент, seed42, AdamW
lr=0,001/weight_decay=0,0001, clipping5, 6400 шагов. TabNet: n_d=n_a=32,
n_steps=3, gamma=1,3, sparsemax, virtual batch128, lambda_sparse=0,001.
Это один стартовый конфиг, не результат настройки TabNet.

Fast2000 выбирает checkpoint. Оставшиеся9534 ленты оцениваются после завершения
обеих моделей и записи frozen.json. Вся selection использовалась исторически;
она не является новым независимым holdout. Late/test не читаются. Один seed
даёт предварительный результат, не оценку устойчивости.

Лимит всей пары —7200с с учётом обучения и промежуточных fast-проверок.
Достижение лимита сохраняет checkpoint и означает незавершённый опыт.
Финальная оценка после фиксации моделей может выйти за этот бюджет.

## Источники и зависимости

MLP-код хранится отдельным замороженным снимком
artifacts/mlp-source-d4f9edb с source.json и SHA каждого Python-файла.
TabNet4.1.0 устанавливается в отдельный target, без изменения общего venv:

```bash
python -m pip install --no-deps --target artifacts/tabnet-deps pytorch-tabnet==4.1.0
python -m unittest discover -s tabnet_research -p 'test_*.py' -v
```

Пример команды для уже проверенного CUDA-хоста:

```bash
python -m tabnet_research.run \
  --mlp-source artifacts/mlp-source-d4f9edb \
  --dependencies artifacts/tabnet-deps \
  --cache PATH_TO_RESEARCH_CACHE \
  --output runs/tabnet/EXP-TABNET-001-r01 \
  --device cuda --steps 6400 --batch-size 16 --seeds 42 --maximum-seconds 7200
```

CLI намеренно требует готовый кеш: он не скачивает данные, не запускает другие
очереди и не создаёт submission. Массивы читаются memmap, в VRAM передаётся
только текущий батч. Перед первым запуском проверяются SHA всех массивов,
кода и сигнатура исходников; конфиг фиксируется и защищён от изменений.

Повтор той же команды продолжает last.pt с optimizer, RNG и курсором.
STOP_AFTER_CHECKPOINT в output сохраняет состояние после очередного шага.
Результаты — status.json, events.jsonl, report.json и отдельные папки моделей.
На сервере команду необходимо запускать через отдельный detached worker или
задачу планировщика; собственная блокировка исключает второй тренер пары.

Первый успешный запуск подтверждается событием batch_completed с CUDA,
конечным loss и изменением шага, а не созданием процесса.

## Проверки

Локально прошли3 синтетических теста: одинаковая инициализация общих PLR/category,
перестановка кандидатов и независимость от меток, градиенты/регуляризация масок,
а также точное продолжение MLP и TabNet после прерывания с optimizer/RNG.
CUDA-ветка проверки устройства выполнится только на доступном GPU.

Upstream low-level TabNet хранит матрицу attention обычным tensor-атрибутом.
Обёртка регистрирует её как buffer для корректного переноса на CUDA и checkpoint.
Это проверка реализации; измеренные результаты сохранены в results/ и tabnet_exp.md.
