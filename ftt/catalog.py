"""Variant definitions; existing configuration dictionaries remain checkpoint-compatible."""

def configuration(number):
    if number in range(21,27):
        config = configuration(17)
        config.update(name=f'FTT-{number:03d}')
        config.update({21:dict(learning_rate=3e-4),22:dict(encoding='ple'),
                       23:dict(score_skip=True),24:dict(dropout=.05),
                       25:dict(depth=3),26:dict(product_token=True)}[number])
        return config
    if number in range(13,21):
        config = configuration(9)
        config.update(name=f'FTT-{number:03d}')
        config.update({13:dict(learning_rate=3e-4),14:dict(learning_rate=3e-5),
                       15:dict(dropout=.05),16:dict(dropout=.2),
                       17:dict(similarity_tokens=True),18:dict(score_skip=True),
                       19:dict(remove_vector_tokens=True),20:dict(context=True)}[number])
        return config
    if number in (9, 10, 11, 12):
        config = configuration({9:1,10:2,11:3,12:4}[number])
        config.update(name=f'FTT-{number:03d}', interactions=True)
        return config
    return dict(name=f'FTT-{number:03d}', width=192 if number == 5 else 96,
                depth=3 if number == 4 else 2, encoding={2: 'plr', 3: 'ple'}.get(number, 'linear'),
                history=number == 7, context=number == 8, loss='bce' if number == 6 else 'softmax')
