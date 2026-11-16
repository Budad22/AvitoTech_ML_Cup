# MLP val-прогнозы для ансамбля

- `mlp_val_predictions.parquet` — вся официальная **val**, 1 389 971 строка и 14 423 ленты.
- `mlp_val_predictions.json` — Run-ID, эпоха, SHA-256 checkpoint/Parquet, полный val NDCG, граница selection/control и commit кода.

Выбран `EXP-024-history-stats-r01`, эпоха 3: MLP на полном train со статистиками
истории пользователя. Критерий выбора — selection NDCG 0,36843792773099354;
полный val и результат сайта для выбора не использовались. Модель не переобучалась.
Полный val NDCG экспортированных logits: **0,367293913557716**.
Граница selection/control: **1790249984**; selection — строго меньше границы.

Схема Parquet: `feed_id:int32`, `item_id:int32`, `score:float32`.
`score` — сырые logits без sigmoid, округления или нормировки. Метки в файл не включены.
Предсказания проверяются на точное покрытие официальных val-ключей, уникальность,
отсутствие пропусков и NaN/Inf; полный NDCG независимо пересчитывается из файла.

Объединять с CatBoost нужно по `(feed_id, item_id)`, а не по позиции строки:

```python
import pandas as pd

mlp = pd.read_parquet("mlp_val_predictions.parquet")
cb = pd.read_parquet("catboost_val_predictions.parquet")
assert not cb.duplicated(["feed_id", "item_id"]).any()
assert len(cb) == len(mlp)
merged = mlp.merge(cb, on=["feed_id", "item_id"], how="outer",
                   suffixes=("_mlp", "_cb"), validate="one_to_one", indicator=True)
assert merged["_merge"].eq("both").all()
```

Подбор веса ансамбля проводить на selection (`timestamp < boundary` из JSON),
поздний control (`timestamp >= boundary`) оставлять для итоговой проверки.
Шкалы score разных моделей могут отличаться; преобразование и смешивание — отдельный
эксперимент, эти файлы содержат исходный выход MLP.

Экспорт: `scripts/export_mlp_val.py`. Скрипт вызывает `evaluate(..., part='all',
output=..., split='val')`. Обычная CLI `predict` для этой задачи не используется.
Веса и исходный датасет в эту папку не включаются. Этот Parquet предназначен для val-ансамбля,
а не для отправки на тест соревнования.
