from paper_code import *

# load the data

a_file = open('/Users/jingyig/Work/Kavli/Data/cleaned_pickle_files/clean_data_johnjohn_210520_s5_distal_sound_reheaded_XYZeuler_notricks.pkl', "rb")
data_raw = pickle.load(a_file)
a_file.close()

ak = list(data_raw.keys())
for i in range(len(ak)):
    print(ak[i])


# prepare the data
data_temp = prepare4ratemap(data_raw)
data = prepare_data4glms(data_temp)
data['file_name'] = 'xxxxxxxx'

# if you want to save the data
# with open('glmdata_kavorka_190620_s4_intermediate_light.pkl', 'wb') as handle:
#     pickle.dump(data, handle)
# and load the saved Pre GLM Data
# a_file = open('glmdata_frank_010620_s1_distal_light.pkl', "rb")
# data = pickle.load(a_file)
# a_file.close()

# do model selection
cell_index = 0
cell_name = data['cell_names'][cell_index]
ms = ForwardSelection(save_file=False)
output_dict, best_model = ms.fit(data, cell_index=cell_index)

f_model = {cell_name: output_dict['%s-keys' % output_dict['best-model']]}

res = calc_backwards_scores('bernoulli', data, cell_index=cell_index, final_model=f_model)




