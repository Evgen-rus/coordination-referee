# Протокол экспериментов (Exp11+)

Документ обязателен для любого нового эксперимента. Он **не меняет** Exp01–Exp10,
baseline, evaluation, data, folds и метрики — он фиксирует то, что уже доказало
пользу в Exp09/Exp10, и добавляет ровно одно правило, которого раньше не было:
**stalled hillclimb**.

Машинная проверка: `python experiments/protocol_guard.py`

---

## 1. Схема одного эксперимента

```
current best
    ↓
ONE hypothesis                      одна основная проверяемая гипотеза
    ↓
exact baseline reproduction          baseline воспроизведён ДО запуска поиска
    ↓
experiment                           параметры зафиксированы ДО просмотра результата
    ↓
honest held-out / cross-fitted       честная проверка, а не in-sample
    ↓
predeclared promotion gate           gate объявлен заранее
    ↓
PROMOTE  →  новая baseline + production build
STOP     →  ничего не создаётся
```

Каждый шаг этого списка уже реализован в Exp09/Exp10 и **переиспользуется**, а не
переписывается. Новый эксперимент импортирует существующий код
(`load_exp07.py`, `reuse.load`, `seal_peaks.py`, `evaluation.metrics`) и повторяет
их структуру.

## 2. Что уже реализовано (переиспользовать, не дублировать)

| Механизм | Где | Что даёт |
|---|---|---|
| строгая верифицированная загрузка | `exp08/load_exp07.py`, `reuse.load` | fail-closed, `ArtifactRejected` фатален, ленивого пути нет |
| sealed window peaks | `exp08/seal_peaks.py` | `peak_pos`/`peak_prob` запечатаны, проверка на каждом использовании |
| reproduction gate | `exp09, exp10` §2 | 5 чисел Exp08 сходятся до 1e-12 **до** любого поиска |
| fast objective == official | `exp09, exp10` §3 | быстрый путь сверяется с `evaluation.metrics` на реальных кандидатах |
| cross-fitted meta-CV | `exp09, exp10` §5 | параметр выбирается на 2 фолдах, применяется к третьему |
| детерминизм | целочисленная сетка Exp09, `k/20` Exp10 | rerun бит-в-бит идентичен |
| promotion gate | `exp09, exp10` §7/§8 | 4 критерия, объявлены заранее |
| тест «нет сборки при FAIL» | `test_blend.py` | `submission_exp10_ab_blend` и `.zip` не должны существовать |

`protocol_guard.py` **не переигрывает** ни один из них. Он только читает уже
записанный вердикт из `results.json` и `results.csv`.

## 3. Правила

1. **Один Exp = одна основная гипотеза.** Побочные находки описываются в
   `RESULTS.md`, но не превращаются в отдельные эксперименты в том же прогоне.
2. **Baseline/current best воспроизводится до запуска поиска.** Числа
   Exp08 (`macro 0.7894323366835861`, `composite 0.7694453917139285`) — это
   reproduction targets, а не ручки настройки. Не сошлось → STOP, ничего не искать.
3. **Параметры, search space и gate фиксируются до просмотра результата.**
   Сетка не расширяется и не уточняется после того, как стало видно число.
4. **Изменение внутри шума или без переносимого эффекта = STOP.** Если дельта
   не переносится на held-out, это не результат, даже если in-sample вырос.
5. **Failed Exp не может стать новым baseline.** Baseline — последний
   промотированный эксперимент; `current_best()` в guard берёт только promote-строки.
6. **Failed Exp не создаёт production build / ZIP / submission.** Guard проверяет
   это по факту: если рядом с STOP-экспериментом нашлась его сборка — нарушение.
7. **Public leaderboard не используется для подбора параметров.** Только как
   reference в отчёте (Exp09/Exp10 это уже делают).
8. **`experiments/results.csv` остаётся основной историей.** Свою строку
   эксперимент получает здесь, а не в отдельном трекере.
9. **Существующие folds, scoring и официальные метрики не трогаются.**
   Валидация — cross-fitted held-out на существующих 3 фолдах Exp07.

## 4. Stalled hillclimb guard

> **Правило.** Если последние 3 последовательных новых эксперимента получили
> STOP / failed promotion gate, следующий эксперимент **не придумывается**.
> Сначала обязателен `root-cause review` последних трёх результатов.

Root-cause review — это файл `experiments/ROOT_CAUSE_REVIEW.md`, который должен
отвечать на все пять вопросов:

| ключ | вопрос |
|---|---|
| `hypotheses` | какие гипотезы проверялись |
| `transfer` | почему они не перенеслись на held-out |
| `headroom` | где остался измеримый headroom |
| `noise` | какие направления уже выглядят noise-dominated |
| `warrant` | есть ли вообще основание для следующего Exp |

Плюс шапка `status: complete` и `covers: [exp09, exp10, exp11]` — список
экспериментов, которые review разбирает. **Просроченный review не подходит:**
если следующий STOP меняет хвост, старый review больше не закрывает новый
stall, и guard его отвергнет.

Это guard для агента, а не исследовательская система: он ничего не генерирует,
не выбирает гипотезу и не планирует эксперимент. Он только отказывает.

### Текущее состояние

| | |
|---|---|
| current best | **exp08** (production build `sub08`) |
| подряд STOP | **2** — exp09, exp10 |
| до блокировки | **1** STOP |
| вердикт | Exp11 **можно** провести как третий эксперимент цикла |

Если Exp11 тоже STOP, то Exp12 не придумывается — сначала review.

## 5. Как провести Exp11

```bash
python experiments/protocol_guard.py                 # 1. проверить, что можно
python experiments/exp11_<name>/run_experiment.py    # 2. reproduction gate первым
python experiments/exp11_<name>/test_<name>.py        # 3. тесты
```

Структура каталога повторяет Exp09/Exp10 (`run_experiment.py`, модуль логики,
`test_*.py`, `RESULTS.md`, `results.json`, `run.log`) — это уже рабочая схема,
копировать её, а не изобретать.

Обязательные STOP-условия, которые останавливают прогон, а не дают число:

- baseline не воспроизвёлся до 1e-12;
- strict-loader rejection;
- fast objective разошёлся с `evaluation.metrics`;
- held-out фолд увидел свои же параметры;
- promotion gate не пройден → **ни сборки, ни ZIP, ни второго метода**.

## 6. Чего делать нельзя

- Не заводить отдельную БД, сервис или orchestration layer.
- Не строить универсальный eval framework — узкая задача, узкий файл.
- Не добавлять скрытый датасет и не дробить текущие данные без доказанной
  необходимости: cross-fitted held-out Exp09/Exp10 остаётся валидацией.
- Не рефакторить Exp01–Exp10 «ради единства» — они зафиксированы.

## 7. Что этот протокол уже сделал

`protocol_guard.py` в репозитории уже проверен и выдаёт:

```
current best / baseline to reproduce : exp08
experiments so far                  : exp01 ... exp10
consecutive STOPS                   : 2 / 3  (exp09, exp10)
root-cause review                  : absent or stale
VERDICT: a new experiment MAY be started.
         1 more STOP(s) and the root-cause review becomes mandatory.
```

`ROOT_CAUSE_REVIEW.md` намеренно **не** создан: Exp11 ещё не существует, и заполнять
review до того, как появился третий результат, значило бы выдумать данные.
