# Transformer — проверка официального бейзлайна

> Актуально: train4 контроль001 завершён (selection0.329372). Ночной full-train002 разрешён и возобновлён после исправления проверки повторных train-кандидатов. Статус и продолжение — docs/WINDOWS_TRAINING_CONTEXT.md.


## EXP-TRF-001 — воспроизведение официального контроля

- Основание: после завершения слабого TabNet006 владелец выбрал переход
  к воспроизводимому сильномуbaseline. Официальныйисточник baseline.zip,
  SHA0bcbed69e4aefb169f05d7dbec080c42c181ec5b587227c64219465b8edde309.
- Архитектура: исходный Ranker изneural_baseline.py, SHA
  87b731df2a5507957c075e93a7f219c1fa4f59b9b7bb9752c59c43087bb2cc06.
  User+candidate-set attention,width64,layers2,heads4,category8,BCE,
  AdamW0.001/0.0001,batch32,5эпох. Полнаярепродукция требует отдельного
  окружения изофициальногоlock (Python3.11,Torch2.10.0,NumPy1.26.4 иостальные).
- Сейчас только подготовка: техническая проверка формы, конечных градиентов,
  памяти/скорости иперестановкикандидатов наRTX3060Ti. ИсходныйклассRanker
  извлекается AST иззакреплённогофайла безизменениякласса; проверка наготовом
  Torch2.8.0 невыдаётся заточное воспроизведениеокружения илиоценкукачества.
- Дляэтойпроверки разрешены6 диагностических train-батчей32 ленты (обычные
  и32самые длинные), веса выбрасываются. Бюджет<2мин, validation/test не читаются.
  Входы собираются изкэшаtrain4: f1..f5, normalizeditem[133:261], missing[517],
  normalizeduser[5:133], category. История и прочиевходыMLP не используются.
- Модельноеобучение/подборпараметров ещё не запущены. После проверкиресурсов
  фиксируетсякоhort, окружение ипротоколcheckpoint наfast, затемтолькоtrain+val.
  Нельзяназыватьtrain4 результатомполногоtrain. СохранённаяMLP — контроль;
  новыйbaseline — сравнениецелыхметодов, не чистаяабляцияархитектуры.
- Общаяметрика — fullNDCG с равнымвесомлент. Late/test неиспользоватьдля
  тюнинга; confirmation ужеисторическииспользовалась. Submission/pushотдельно.


### EXP-TRF-001 — техническая проверка на Windows прошла

- Прочитан официальный Ranker (SHA закреплёнвыше), выполнены6 диагностических
  optimizer-батчей наtrain4. Только train; validation/test не читались,
  веса не сохранялись. Это не завершённый модельный опыт и не NDCG.
- Windows: RTX3060Ti8GiB, Torch2.8.0+cu126/CUDA12.6. Параметров84841.
- Обычныйбатч32ленты:2285кандидатов,max263,3шага0.375с с первымCUDAпрогревом,
  peak100541440байт.32самыедлинныеленты:9428кандидатов,max299,
  3шага0.063с после прогрева,peak111289344байт (≈106.1MiB).
- Перестановка кандидатов совместно с признаками переставляет оценки;
  численная проверкаatol/rtol2e-5 прошла. Конечныеloss/градиенты подтверждены.
- Исходныйкласс модели не менялся; зависимыеCLI/Polars части не исполнялись.
  Официальноеокружение (Torch2.10.0/NumPy1.26.4) пока не воспроизведено.
- Отчёт: artifacts/official-baseline-review/transformer-preflight.json
  вдеревеTabNet; скрипт artifacts/official_transformer_preflight.py.
  Windows C:\AvitoBaselines\official-check-001. Долгойзадачиbaseline нет.
- Вывод: сама модель помещается в8GiB с большимзапасом. Следующийресурсный
  вопрос — hostRAM/загрузчикполногоtrain. Нынешний519-мерныйкэшMLP
  не следует линейнораздувать на512шардов; хранить исходныеfloatпризнаки,
  user-вектор одинразналенту, int8item-memmap ииндексы, формироватьтекущийбатч.
- Экстраполяциятолько прогретыхGPUбатчей:492380батчей для5эпох по3151208лент
  /32 ×0.021с ≈2.87ч. Это оптимистичныйориентирбезоценкиполногоIO,
  подготовки ивалидации, не обещаннаядлительностьобучения.


## EXP-TRF-001-r01 — первый модельный контроль, разрешён запуск

- Новая команда владельца: «работа опять на компе будет? запускай».
- Windows RTX3060Ti, отдельный C:\AvitoBaselines\EXP-TRF-001-r01.
  Официальноеокружение устанавливается изuv.lock сCUDA-сборкойTorch2.10.0+cu128;
  venvTabNet не изменяется. ЗаменаCPU/универсальногоwheel наCUDA записывается.
- ИсходныйRanker иcollate_feeds иззакреплённогоneural_baseline.py не меняются.
  Адаптер читает нормализованныевходы изстарогоcachetrain4, вместоParquet/memmap
  официальногоDataset. Дообучение/история/PLR неиспользуются. Это адаптированный
  контроль официальноймодели, не буквальныйзапускCLI на полномtrain.
- train4:24592ленты/1925512кандидатов, seed42,5эпох,batch32,3845шагов.
  BCE по всемреальнымкандидатам, AdamWlr0.001/weight_decay0.0001, безAMP/клиппинга.
  Checkpoint по минимальномуrowLogloss наfast2000, как критерий официальногокода.
  Основнойотчёт — fullNDCG по лентам. После5эпох frozen.json, затемselection
  иconfirmation9534. Историческииспользованнаяval независимой не является.
- Нормализованныеuser/item берутся изкэша сfloat32; возможныULP-отличия
  отF.normalize оригинальногоDataset. Отличиязагрузчика/когорты/валидации
  явнофиксируются. Предсказания — float32sigmoid, как вofficialpredict.
- Время ориентировочно10–20мин вместе с новымокружением; самообучение ожидается
  несколькоминут по техническомуGPU-замеру. Это прогноз, не измеренныйрезультат.
  Лимитtrain1800с, отдельнаязадачаAvitoTransformer001 с90мин наокружение+train.
- Готовится точноеresume модель/optimizer/RNG/курсорDataLoader безизменения
  прежнихмоделей. Test/submission/push не выполняются. Итоговыхметрик пока нет.


### EXP-TRF-001-r01 — обучение подготовлено, Windows-проверки прошли

- Добавлен transformer_research/: неизменённыеRanker/collate черезAST,
  адаптерCachedFeeds и5-эпоховыйBCEtrainer сcheckpointпоfastLogloss.
- NativeDataLoadershuffle сохраняется. Приresume восстанавливаются epoch RNG,
  iterator/cursor, globalCPU/CUDA RNG, model иoptimizer; синтетическая проверка
  interrupted→resume совпала с непрерывнымобучением по весам/optimizer/истории.
-4 проверки наWindows Python3.11/Torch2.8 прошли за1.740с; перед modeltraining
  они повторяются наофициальномTorch2.10. НаMac3.9 триконтрактаусловноskipped
  из-заzip(strict=True) официальногоcollate; проверкаметрики прошла.
- ОтдельнаязадачаAvitoTransformer001 уже устанавливаетофициальноеокружение
  поuv.lock черезuv0.13.9. CUDAwheel берётся изофициальногоcu128index.
  GPU-модельныйзапуск ожидает завершенияустановки; долгихдругихочередей нет.
- КоддоставленZIPSHA4c5493758401abb855eac0f1fc57b76fc89aa650121ee7af9a0b5a8f93eb58fb.
  SOURCE_READY разрешает автостарт только после пройденныхконтрактов.
  Результатовкачества пока нет, test/submission/push не выполнялись.


### EXP-TRF-001-r01 — реальное обучение наTorch2.10 подтверждено

- Официальные зависимости изuv.lock установлены в отдельныйvenv, CUDAwheel
  torch2.10.0+cu128 установлен отдельно. `uv pip check`: все пакеты совместимы.
-4 контракта повторно прошли уже вновомокружении за3.438с, включая CUDAbackward
  иexactresume CPUoptimizer/RNG/nativeDataLoader.
- РеальныйtrainerPID43084, device=cuda, событиеcompleted,
  прогрессepochNone/5 stepNone/3845.
  Скорость около45шагов/с, оценкасамогообучения≈2мин; установказаняла основное
  времявпервомконтроле. Это не оценка полногоtrain.
- Отчёт ещё ожидается. Первыйконтроль по-прежнемуtrain4, безновыхгипотез
  сверхофициальнойархитектуры/настроек. Никакихtest/submission/push.


## EXP-TRF-001-r01 — первый адаптированный контроль завершён

- Пятьэпох/3845шагов выполнены безошибок наtrain4. Завершён23:49:57МСК.
  Trainerwall92.969с с финальнойоценкой; fit 88.234с.
  Установкаокружения считаетсяотдельно. GPUpeak137506304байта≈131.1MiB.
- Checkpoint выбран поfastrowLogloss, epoch4: fastLogloss0.0742208033646525.
  ИтогselectionNDCG0.329371524620737, fastNDCG0.3308910220814823,
  confirmationNDCG0.3290527712201191. Это не результатполногоtrain/сайта.
- ОтносительносохранённойMLP наэтойкогорте selectiondelta
  -0.031962211235754. Этотконтрольне заменяетMLP.
  Различаютсявходы/функцияпотерь/epoch/batch; это сравнениеготовыхметодов,
  не чистый тестархитектуры. Никакого выводаопределеTransformerнаполномtrain.
- FastNDCGepoch5=0.3324196726051936 численно выше, но исходныйкритерийLogloss
  выбралepoch4. Решение не менялось после открытияconfirmation. Возможная
  сменакритерияcheckpoint требует отдельнойгипотезы, не делаем заднимчислом.
- Независимо пересчитаныper-feedNDCG всех11534лент/1101295строк, совпали до1e-12;
  source/cache/codeSHA иfrozen/checkpoint/критерийвыбора проверены.
  Отчёт transformer_research/results/EXP-TRF-001-r01.json, сырыеотчёты/прогнозы
  artifacts/windows-results/EXP-TRF-001-r01 (игнорируютсяGit).
- Окружение поофициальномуuv.lock: Torch2.10.0+cu128,NumPy1.26.4,
  Polars1.14.0,Click8.1.8,Loguru0.7.3,CatBoost1.2.10.4контракта прошли.
  КлассыRanker/collateне менялись; адаптацияDataset/валидацииописана выше.
- Вывод: официальныйконтроль на4/512шардах измерен и техническиготов.
  Дальше нужен крупный/полныйtrain скомпактнымзагрузчиком и замеромRAM/IO;
  не продолжать десятки мелких настроек поэтоймалойвыборке.
- Полныйtrain ещё не запущен. Test/submission/push не выполнялись.


## EXP-TRF-002 — полный train, один ночной запуск

- Владелец сначала предложилполовину, затем явно изменилрешение: «хотя давай
  на полную, раз так долго, на ночь оставлю». Текущаякоманда — весьtrain512шардов,
 5эпох, исходнаяархитектура/optimizer/BCE/batch32/seed42 как001, снуля.
  Половинныйзапуск не создавался. Эксперимент EXP-TRF-002-r01.
- Нужны компактныеmemmap-признаки: f1..f5, категории, target, itemindex/keys,
  user-вектор одинразнаfeed; int8itemматрицы отдельно. Никаких519float
  накаждогокандидата. Rawисходники/read-onlyстарыеcache/checkpointне меняются.
- НаWindows переносятсяполныеtrain/valitemматрицы, локальные64trainшарда,
  valтаблицы. Остальные448trainшардов компьютер докачает напрямую изофициального
  ZIP черезограниченныеHTTPRange, сETag/CRC/SHA. Testчлены не извлекаются.
- Validationконтракт прежний: selection11534/1101295,boundary1790249984,
  fast2000/confirmation9534. Категорииfitтолькоtrain. Послеfreeze финальнаяоценка.
- ОтдельныйrootC:\AvitoBaselines\EXP-TRF-002-r01, задачаAvitoTransformer002,
  datasetC:\AvitoBaselines\datasets\full-v1. Готовыйvenv001 читаетсябезизменений,
  PyArrow21 дляподготовки ставится влокальныйtargetdependencies.
- Предварительнаяоценка3–6чобучения, плюсзагрузка/подготовка. Замеритьреальный
  compactloaderпередначалом дляуточнения. Ограничениеtrain8ч иобщаязадача12ч
  — предохранители, не обещаниекачества/сходимости. Step-budget≈492380.
- Требуютсяпроверкисоответствиявходов/ключей/масок/val иexactresume сбыстрым
  пропускоминдексовnativeSampler (не перечитыватьпройденныебатчи).
- Сейчасподготовка. Никакихитоговыхметрик002ещёнет;test/submission/push нет.

### EXP-TRF-002-r01 — фоновая подготовка запущена

- Задача AvitoTransformer002 Running; стадия downloading_full_train.
  12 034 325 865 байт локальных исходников перенесены за 147.79 с; RAW_TRANSFER_READY
  создан. После сетевой загрузки Windows самостоятельно проверит SHA всех файлов.
- Код компактного загрузчика и полного pipeline доставлен; 6 контрактов на
  Windows/Python3.11/Torch2.10/CUDA прошли за 5.903 с, включая преобразование
  Parquet, пропуск sampler без чтения данных, exact resume optimizer/RNG/weights,
  padding/перестановки, CUDA backward и пессимистические ties NDCG.
- SOURCE_READY разрешает автоматическое продолжение после download: build →
  проверка всех 11534 selection-входов против общего cache → 40 пробных GPU-батчей
  с отбрасыванием весов → новый seed42 full train 5 эпох.
- На момент записи основное обучение ещё не началось; NDCG002 не измерен.

- Preflight на реальных данных остановился до GPU: проверка размера train int8
  матрицы переполняла np.prod с Windows int32 (10.44GB). Исправлено явным
  int64 произведением; повреждения данных нет, SHA всех 12.03GB совпали.
  Основной trainer ещё не начинался; проверка повторяется.

### EXP-TRF-002-r01: real-data compact preflight passed

- All 12.03 GB transferred files matched SHA256. The int64 matrix-size fix
  passed on Windows; the earlier preflight failure occurred before training.
- Diagnostic cache: 4 shards, 24538 feeds / 1915387 candidates; selection
  11534 feeds / 1101295 candidates. Build took 2.812 seconds. All selection
  keys/offsets/targets and native inputs matched the common cache; max error 0.0.
- 40 disposable CUDA training batches took 1.187 seconds, 33.6984 batches/sec;
  peak GPU 110569472 bytes. Diagnostic weights discarded. Extrapolated full
  training: 492380 steps, 4.059 hours plus download/preparation. This is an
  estimate from a short profile, not a measured full-run duration.
- Full-data profile repeats automatically before the production trainer.
  AvitoTransformer002 Running, SOURCE_READY present, download continues.
  Production training has not started yet; no EXP-TRF-002 metrics measured.
- Profile: artifacts/full-loader-preflight.json, Windows preflight/loader-profile.json.
  Automatic chain: download CRC/SHA -> build -> audit -> 5 epochs full train.
  No test inference, submission, or push. Laptop no longer required.

### EXP-TRF-002-r01: production full train started on CUDA

- Full512 verified: 3151208 feeds /246745836 candidates.
- Build stopped01:04:39MSK on an excessive duplicate train-key check. Raw feed
  1233995 contains item9455009 at positions128/129, targets0/0, embedding index
  70119948. Native CandidateData preserves these rows. Fixed: preserve/report
  train duplicates without dedup; reject val duplicates. Regression covered.
- All6 target Windows contracts passed2.080sec; task restarted about01:07MSK.
- Full compact ready; all11534 selection inputs exactly match old cache (error0).
  Full profile40CUDA batches1.485sec,26.9360batch/sec,peak110657536bytes.
  Estimated trainer duration5.0777h. Disposable profiling weights discarded.
- Production trainer PID22540 started01:11:13MSK. CUDA optimizer steps1/20
  confirmed, finite loss, epoch1/5,total492380steps. No final NDCG yet.
- Config/status/profile/compact metadata snapshot saved in
  transformer_research/results/EXP-TRF-002-r01-start.json (snapshot_only=true).
  Resume/host context: docs/WINDOWS_TRAINING_CONTEXT.md.
- No test inference, submission or push. Do not mutate live trainer source.
