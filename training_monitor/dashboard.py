from datetime import datetime, timedelta, timezone
from html import escape

MSK = timezone(timedelta(hours=3))
STAGES = {"train": "Обучение", "evaluate": "Валидация", "train-a": "Стадия A",
          "train-b": "Стадия B", "val": "Валидация"}
METRICS = {"a": "Стадия A", "b": "Стадия B", "baseline": "Бейзлайн",
           "listwise_1epoch": "Listwise · 1 эпоха", "bce_step186200": "BCE · шаг 186 200",
           "bce_5epochs": "BCE · 5 эпох"}


def clock(value):
    return datetime.fromisoformat(value).astimezone(MSK).strftime("%d.%m · %H:%M МСК")


def job_card(job):
    title = escape(job["id"])
    lines = []
    p = job.get("progress", {})
    if job["status"] == "running":
        lines.append(escape(STAGES.get(job["stage"], job["stage"] or "Запуск")))
        if p.get("percent") is not None:
            lines.append(f'<strong>{p["percent"]:.1f}%</strong><progress aria-label="Прогресс" max="100" value="{p["percent"]}"></progress>')
            lines.append(f'Шаг {p["step"]:,} из {p["steps"]:,}'.replace(",", " "))
        if p.get("estimated_stage_end"):
            lines.append("Окончание этапа ≈ " + clock(p["estimated_stage_end"]))
        if job["next_stages"]:
            lines.append("Далее — " + ", ".join(escape(STAGES.get(s, s)) for s in job["next_stages"]))
    for key, score in (job.get("ndcg") or {}).items():
        lines.append(f'{escape(METRICS.get(key, key))} <b>{score:.5f}</b>'.replace(".", ","))
    return f'<article><h3>{title}</h3>' + ''.join(f'<p>{line}</p>' for line in lines) + '</article>'


def render(data):
    if data is None:
        current = 'Сервер временно не получил статус. Повтор через 15 секунд.'
        queue = finished = updated = ''
    else:
        current = ''.join(map(job_card, data["current"])) or 'Сейчас обучение не идёт.'
        states = {"waiting":"ожидает", "queued":"в очереди Windows", "blocked":"заблокировано",
                  "awaiting_start":"готово к запуску"}
        queue = ''.join('<p>' + escape(j["id"]) + ' · ' + escape(states.get(j["state"], j["state"]))
                        + (' после ' + escape(j['after']) if j['after'] else '') + '</p>'
                        for j in data["queue"]) or 'Очередь пуста.'
        finished = ''.join(job_card(j) for j in data["jobs"] if j["status"] == "completed") or 'Пока нет результатов.'
        updated = 'Обновлено ' + clock(data["updated_at"])
        for j in data["jobs"]:
            if j["status"] in ("failed", "stopped"):
                current += '<p>' + escape(j["id"]) + ' · ' + ('ошибка' if j["status"] == 'failed' else 'остановлено') + '</p>'
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15"><title>Монитор обучения</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f4;color:#172e2c;font:16px/1.5 Inter,system-ui,sans-serif}}main{{max-width:48rem;margin:auto;padding:clamp(18px,5vw,52px)}}h1{{font-size:clamp(30px,5vw,42px);margin:0}}small{{color:#49605a}}section{{background:white;border:1px solid #d4e1dc;border-radius:14px;margin:18px 0;padding:20px}}h2{{font-size:20px;margin:0 0 10px}}h3{{font-size:16px;overflow-wrap:anywhere}}p{{margin:5px 0}}article+article{{border-top:1px solid #dbe4e1;margin-top:14px}}strong{{font-size:28px}}progress{{width:100%;accent-color:#19725d}}b{{font-variant-numeric:tabular-nums}}</style>
<main><h1>Монитор обучения</h1><small>{updated}</small><section><h2>Сейчас</h2>{current}</section>
<section><h2>Очередь</h2>{queue}</section><section><h2>Результаты · NDCG</h2>{finished}</section></main></html>'''
