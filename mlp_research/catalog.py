"""Predeclared hypotheses, matched controls, and bounded promotion by family."""


def hypotheses():
    base = dict(family='mlp', backbone='mlp', history=True, context=False, encoding='raw',
                width=256, depth=2, dropout=.1, activation='relu', layer_norm=False,
                learning_rate=.001, weight_decay=.0001, loss='bce', frequencies=16,
                embedding=16, sigma=1., frozen=False, shared=False, raw_skip=False,
                scalar_history=False, normalize='raw', bins=16)
    cases = []

    def add(name, description, **changes):
        cases.append(dict(base, name=name, description=description, **changes))

    add('baseline', 'Обычная MLP без истории', family='control', history=False)
    add('history_control', 'Контроль: MLP со статистиками истории', family='control')
    for name, scale in [('001', .01), ('01', .1), ('1', 1.), ('3', 3.), ('10', 10.)]:
        add('plr_sigma_' + name, f'PLR: начальный масштаб частот {scale}',
            family='plr', encoding='plr', sigma=scale)
    for count in (8, 32, 64):
        add(f'plr_k{count}', f'PLR: {count} частот на признак', family='plr', encoding='plr', frequencies=count)
    for dimension in (8, 32):
        add(f'plr_d{dimension}', f'PLR: вектор числового признака {dimension}D',
            family='plr', encoding='plr', embedding=dimension)
    for name, description, changes in [
        ('plr_lite', 'PLR с общей проекцией признаков', dict(shared=True)),
        ('plr_frozen', 'PLR с фиксированными частотами', dict(frozen=True)),
        ('plr_raw_skip', 'PLR вместе с исходными числами', dict(raw_skip=True)),
        ('plr_standard', 'PLR после стандартизации по train', dict(normalize='standard', sigma=.1)),
        ('plr_quantile', 'PLR после квантильного преобразования по train', dict(normalize='quantile')),
        ('plr_history', 'PLR для f1–f5 и 17 числовых статистик истории', dict(scalar_history=True, normalize='standard', sigma=.1)),
        ('plr_low_lr', 'PLR с меньшим шагом обучения', dict(learning_rate=.0003)),
    ]:
        add(name, description, family='plr', encoding='plr', **changes)
    add('numeric_linear', 'Числовые эмбеддинги Linear → ReLU', family='encoding', encoding='linear')
    add('numeric_ple8', 'Кусочно-линейные числовые эмбеддинги, 8 интервалов', family='encoding', encoding='ple', bins=8)
    add('numeric_ple32', 'Кусочно-линейные числовые эмбеддинги, 32 интервала', family='encoding', encoding='ple', bins=32)
    add('numeric_rbf', 'Числовые эмбеддинги на гауссовых функциях', family='encoding', encoding='rbf')
    add('plr_softmax', 'PLR с listwise softmax вместо BCE', family='plr', encoding='plr', sigma=.1, loss='softmax')
    add('mlp_softmax', 'MLP с историей и listwise softmax', loss='softmax')
    add('mlp_mixed_loss', 'MLP с BCE + 0,1 × listwise softmax', loss='mixed')
    for name, description, changes in [
        ('mlp_small', 'Меньшая MLP: 128 → 64', dict(width=128)),
        ('mlp_wide', 'Широкая MLP: 512 → 256', dict(width=512)),
        ('mlp_deep', 'MLP с четырьмя скрытыми слоями', dict(depth=4)),
        ('mlp_gelu', 'MLP с GELU', dict(activation='gelu')),
        ('mlp_silu', 'MLP с SiLU', dict(activation='silu')),
        ('mlp_no_dropout', 'MLP без dropout', dict(dropout=0.)),
        ('mlp_dropout03', 'MLP с dropout 0,3', dict(dropout=.3)),
        ('mlp_layernorm', 'MLP с LayerNorm скрытых слоёв', dict(layer_norm=True)),
        ('mlp_low_lr', 'MLP с learning rate 0,0003', dict(learning_rate=.0003)),
        ('mlp_high_lr', 'MLP с learning rate 0,003', dict(learning_rate=.003)),
        ('mlp_decay', 'MLP с weight decay 0,01', dict(weight_decay=.01)),
        ('mlp_context', 'MLP со статистиками остальных кандидатов ленты', dict(context=True)),
        ('mlp_residual', 'MLP с остаточными блоками', dict(backbone='residual')),
        ('mlp_gated', 'MLP с обучаемыми воротами GLU', dict(backbone='gated')),
    ]:
        add(name, description, **changes)
    add('cross_network', 'Два слоя низкоранговых пересечений + MLP', family='alternative', backbone='cross')
    add('feature_transformer1', 'Один Transformer-слой между группами признаков', family='alternative', backbone='transformer', depth=1)
    add('feature_transformer2', 'Два Transformer-слоя между группами признаков', family='alternative', backbone='transformer', depth=2)
    add('linear_ranker', 'Линейное ранжирование тех же признаков', family='alternative', backbone='linear')
    assert len({case['name'] for case in cases}) == len(cases)
    return cases


CONTROLS = ('baseline', 'history_control')


def promote(results, catalog):
    """Keep family coverage and explicit slow learners instead of a global top-k."""
    by_name = {case['name']: case for case in catalog}
    successful = {result['name']: result for result in results if result['status'] == 'completed'}
    chosen = [name for name in CONTROLS if name in successful]
    for family, count in [('plr', 4), ('encoding', 2), ('mlp', 2), ('alternative', 2)]:
        ranked = sorted((r for r in successful.values() if by_name[r['name']]['family'] == family),
                        key=lambda result: result['fast_ndcg'], reverse=True)
        chosen += [result['name'] for result in ranked[:count]]
    for slow in ('plr_low_lr', 'mlp_low_lr', 'feature_transformer2'):
        if slow in successful:
            chosen.append(slow)
    return list(dict.fromkeys(chosen))
