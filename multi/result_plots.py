import TSMG

base_fn_1 = '_MET_True_ndraws_'
base_fn_2 = '_L_2500_ALPHA_VEC_'
base_fn_3 = '_NUM_CORES_4_THETA_DIM_4_PARAMETRIC_'
base_fn_4 = '_MULTIPLE_'
base_fn_5 = '_CT_SEEDS_'
base_fn_6 = '_start_seed_3_ELLS_6'
defalpha = '[0-5-0-5-0-5-0-5]'
defmus = '0,0,0,0'

fig2_dict = {'2': [], 'p': [], 'b': [], 'g': []}
fig4_dict = {'2': [], 'p': [], 'b': [], 'g': []}
mult_dict = {'2': '3', 'p': '3', 'b': '3', 'g': '0-375'}

for game_id in ['g']:#['2', 'p', 'b', 'g']:
    fn_2 = base_fn_1 + str(TSMG.draws_dict[game_id]) + base_fn_2
    # Nonparametric results
    fn_2n = 'N_30' + fn_2 + defalpha + '_MUS_' + defmus + base_fn_3 + 'False'
    fn_2n += base_fn_4 + mult_dict[game_id] + base_fn_5 + '10' + base_fn_6
    # Parametric
    fn_2p = 'N_30' + fn_2 + defalpha + '_MUS_' + defmus + base_fn_3 + 'True'
    fn_2p += base_fn_4 + mult_dict[game_id] + base_fn_5 + '10' + base_fn_6
    # Self-play, default prior
    fn_4 = 'N_20' + fn_2 + defalpha + '_MUS_' + defmus + base_fn_3 + 'False'
    fn_4 += base_fn_4 + mult_dict[game_id] + base_fn_5 + '1' + base_fn_6
    # Self-play, alt prior
    if game_id == '2':
        fn_4a = 'N_20' + fn_2 + '[2---0-5-0-5-0-5]_MUS_' + defmus + base_fn_3 
    elif game_id == 'g':
        fn_4a = 'N_20' + fn_2 + defalpha + '_MUS_0,2,0,0' + base_fn_3
    else:
        fn_4a = 'N_20' + fn_2 + '[0-5-0-5-2---0-5]_MUS_' + defmus + base_fn_3 
    fn_4a += 'False' + base_fn_4 + mult_dict[game_id] + base_fn_5
    fn_4a += '1' + base_fn_6
    fig2_dict[game_id] = [fn_2n, fn_2p]
    fig4_dict[game_id] = [fn_4, fn_4a]


for game_id in ['g']:#['2', 'p', 'b', 'g']:
    TSMG.plot_comparison(game_id, fig2_dict[game_id][0], fig2_dict[game_id][1],
      30, save=game_id+'_comparison', thresholds=[1.5, 3, 4], num_taus=5)#10)
    TSMG.plot_selfplay(game_id, fig4_dict[game_id][0], fig4_dict[game_id][1],
      20, save=True)
