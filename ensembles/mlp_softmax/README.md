# Val-прогнозы MLP с listwise softmax

Полные прогнозы находятся в Git в архиве
[artifacts/mlp_softmax_val_predictions.zip](../../artifacts/mlp_softmax_val_predictions.zip).
Он содержит два файла с путями `ensembles/mlp_softmax/`:

- `mlp_val_predictions.parquet` — все 1 389 971 пары, 14 423 ленты,
  `feed_id:int32`, `item_id:int32`, `score:float32`.
- `mlp_val_predictions.json` — происхождение checkpoint, хеши и результаты аудита.

Распаковка из корня репозитория:

```powershell
Expand-Archive -LiteralPath artifacts/mlp_softmax_val_predictions.zip -DestinationPath . -Force
```

Прогнозы — сырые logits без sigmoid/softmax, округления и нормировки.
Checkpoint: `EXP-024-softmax-r01`, эпоха 4; код обучения:
`47368f0ad37ef1d7516779606e571f3f1db966ed`.
Полный val NDCG: **0,3620791143318138**. Переобучение и test не запускались.

SHA-256 архива:
`b249bc47e81b4934f61a6245bf5b1ee5154b2cc133bdfcbf55e55aeb6a439650`.
SHA-256 Parquet:
`68b35bc719293b264dbebc9e8b91f28f52977ac8c600aa8f013da2a2b37d4d0e`.

Смешивать с другими моделями по `(feed_id, item_id)`, не по порядку строк.
Данные и веса модели в архив не включены. Экспорт history_stats находится отдельно
в `ensembles/mlp/` и сохранён без изменений.
