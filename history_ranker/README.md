# EXP-TRF-003

Код реализует архитектуру и эксперимент из [спецификации](../docs/YO_RANKER_HANDOFF.md). Стадия A обучается с отдельной таблицей поправки на позицию показа. Стадия B начинает с последнего checkpoint A и использует полный список пар положительных и отрицательных кандидатов в каждой ленте. Итоговый evaluator требует оба зафиксированных checkpoint и проверяет связь B с A.

Команды ниже рассчитаны на PowerShell в папке `C:/AvitoBaselines/EXP-TRF-003-r01`. В эту папку нужны `history_ranker/`, `recsys_data/` и `tests/` из одного состояния ветки. Исходные данные и прежний компактный кэш читаются на месте.

```powershell
$base = 'C:/AvitoBaselines'
$previous = "$base/EXP-TRF-002-r01"
$py = "$base/EXP-TRF-001-r01/.venv/Scripts/python.exe"
$env:PYTHONPATH = "$previous/dependencies"
$raw = "$base/datasets/full-v1/raw"
$compact = "$base/datasets/full-v1/compact"

& $py -m unittest discover -s tests -v
& $py -m history_ranker prepare --scope smoke --raw $raw --compact $compact --output cache-smoke
& $py -m history_ranker smoke --cache cache-smoke --output profile-local
& $py -m history_ranker pilot --cache cache-smoke --output runs/train4-a --microbatch 8
& $py -m history_ranker evaluate-pilot --cache cache-smoke --pilot runs/train4-a --output runs/train4-early
```

Автономный контроль на ПК выполняется заданием `AvitoHistoryRanker003Pilot` через `pilot_windows.ps1`. Состояние записывается в `runs/train4-a/status.json`, журналы находятся в `pilot.log` и `pilot.err.log`. Прерванный запуск продолжает `last.pt` при повторе той же команды с теми же данными, версией кода и параметрами. Поле `elapsed_seconds` в `status.json` суммирует время в цикле обучения по всем попыткам, включая повтор шагов после последнего checkpoint. Простой между запусками исключён. После аварийного выключения точность ограничена последней записью статуса или checkpoint. Повтор завершённой команды с той же конфигурацией проверяет контрольную сумму и сохраняет исходный статус. Профиль измеряет дисковые чтения, подготовку батча, forward, backward и шаг оптимизатора. Скорость зависит от того, какая часть матрицы эмбеддингов осталась в файловом кэше ОС.

Ниже команды для полного опыта после отдельного задания владельца. Каждую команду запускать после успешного завершения предыдущей. Для многочасового обучения использовать отдельную задачу Windows с достаточным лимитом времени.

```powershell
& $py -m history_ranker prepare --scope full --raw $raw --compact $compact --output cache-full
& $py -m history_ranker smoke --cache cache-full --output profile-full --steps 200
& $py -m history_ranker train-a --cache cache-full --output runs/full-a --microbatch 8
& $py -m history_ranker train-b --cache cache-full --stage-a runs/full-a --output runs/full-b --microbatch 8
& $py -m history_ranker evaluate --cache cache-full --stage-a runs/full-a --stage-b runs/full-b --baseline "$previous/runs/EXP-TRF-002-r01/best.pt" --official "$previous/official/neural_baseline.py" --output runs/full-late
```

Проверка пилота использует 22–23 сентября UTC. Итоговое сравнение готовых моделей использует 24 сентября UTC. В `report.json` сохраняются NDCG и сверка с CatBoost, для итогового сравнения ещё и интервалы парных разностей бутстрапом по пользователям. Ключи кандидатов и прогнозы сохраняются в `scores.parquet`.
