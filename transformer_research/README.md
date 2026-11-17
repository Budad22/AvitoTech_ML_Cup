# Официальный candidate-set Transformer: адаптированный контроль

EXP-TRF-001-r01 использует неизменённые `Ranker` и `collate_feeds` из
официального `neural_baseline.py` (SHA закреплён в `core.py`). Обучение:
5 эпох, batch32, seed42, BCE по реальным кандидатам, AdamW0.001/0.0001,
без истории, PLR, AMP и clipping. Checkpoint выбирается по row Logloss fast2000.
После заморозки решения считается полный NDCG на selection11534 и confirmation9534.

Этот запуск — **train4**, не полный train. Вместо исходного Dataset читается
уже проверенный кэш: f1–f5, нормализованный item128, missing-флаг,
нормализованный user128 и категория. Возможны ULP-отличия от нормализации
исходных int8 через PyTorch. Словарь и нормализация происходят из train.
Score — float32 sigmoid, как в официальном predict. Target и ID не входят
в признаки. Внешняя val исторически использовалась; независимость не заявляем.

Windows: `C:\AvitoBaselines\EXP-TRF-001-r01`, задача `AvitoTransformer001`.
Окружение отдельное: официальный uv.lock и CUDA wheel torch2.10.0+cu128.
Код извлекает только два исходных определения через AST после проверки SHA;
официальные CLI/чтение test не исполняются. Существующие MLP/TabNet не меняются.

До обучения пройдены на Windows четыре контракта: full NDCG/ничьи,
адаптер/padding/перестановки, конечные CUDA-градиенты и точное CPU-resume
с native DataLoader, optimizer и RNG. В новом окружении они повторяются.

Результаты `runs/EXP-TRF-001-r01`: config/status/events, best/last checkpoint,
fit-result.json, frozen.json, report.json и selection_scores/ndcg.npy.
Checkpoint сохраняется каждые100шагов, в конце эпохи и по STOP_AFTER_CHECKPOINT.
Повтор той же команды продолжает сохранённый шаг. Конфиг/код/данные защищены SHA.
Лимит обучения1800с: достижение означает incomplete_budget, не отрицательный опыт.
Установка+обучение живут в отдельной задаче; отключение SSH не останавливает её.

Test и submission в этом trainer отсутствуют. Публикация GitHub только после
локального diff и явного одобрения владельца.

## Полный ночной train: EXP-TRF-002-r01

По явной команде владельца: все 512 train-шардов, 5 эпох с нуля, те же
Ranker/collate/BCE/AdamW/batch32/seed42. `fetch.py` получает только train
из закреплённого ZIP (ETag/CRC/SHA), повторно использует 64 локальных шарда.
`build_compact.py` делает два прохода Parquet с проекцией колонок; хранит f1–f5,
индексы item, категории, метки, ключи и один int8 user128 на ленту. Полная
int8 item-матрица читается memmap; F.normalize выполняется как в исходном Dataset.
Это позволяет обойти исходное хранение всех candidate-тензоров в RAM.

Контрольная val прежняя: 11534 ленты/1101295 кандидатов, fast2000/confirmation9534.
Ключи/offsets/метки проверяются целиком; `audit_compact.py` сравнивает входы
всех selection-лент с прошлым cache и измеряет 40 пробных CUDA-батчей, веса
которых выбрасываются. Производственные веса и RNG создаются заново в trainer.

`compact.shuffled_loader` оборачивает исходный BatchSampler/RandomSampler;
при resume пропускаются индексы без чтения пройденных feed. Тест сравнивает
веса, optimizer, RNG и историю с непрерывным исходным DataLoader побитно.

Запуск Windows: `transformer_research/full_windows.ps1`, задача
`AvitoTransformer002`, root `C:\AvitoBaselines\EXP-TRF-002-r01`.
Данные в `C:\AvitoBaselines\datasets\full-v1`, исходный venv001 читается
без изменений; Arrow21 установлен отдельно в dependencies через --target.
Pipeline скачивает и проверяет данные, проверяет тесты, строит compact cache,
проверяет входы/скорость и запускает train. Основной запуск ограничен 8 часами,
вся задача 12 часами. Достижение лимита сохраняет last.pt и означает incomplete.

```powershell
python -m transformer_research.train --source official/neural_baseline.py `
  --cache C:/AvitoBaselines/datasets/full-v1/compact --dataset-format compact `
  --experiment EXP-TRF-002-r01 --output runs/EXP-TRF-002-r01 `
  --device cuda --epochs 5 --batch-size 32 --maximum-seconds 28800
```

Отключение ноутбука не останавливает задачу. Проверка состояния:

```powershell
Get-ScheduledTask -TaskName AvitoTransformer002
Get-Content C:\AvitoBaselines\EXP-TRF-002-r01\pipeline-status.json
Get-Content C:\AvitoBaselines\EXP-TRF-002-r01\runs\EXP-TRF-002-r01\status.json
```
