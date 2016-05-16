import heapq

data_file = '../../../data/test.full'

write_out_dir = '../data/'
attention_ppl = write_out_dir + 'test_ppl_one_weight'
lstm_ppl = write_out_dir + 'test_ppl_lstm' 
good_file = write_out_dir + 'good_test.full' 
bad_file = write_out_dir +  'bad_test.full'
log_file = write_out_dir +  'performance_comparison.log'
extreme_number = 20

with open(attention_ppl,'rb') as f_read :
	all_ppl_str = f_read.read().splitlines()
	all_ppl_attention = [float(ppl_str) for ppl_str in all_ppl_str]

with open(lstm_ppl,'rb') as f_read :
	all_ppl_str = f_read.read().splitlines()
	all_ppl_lstm = [float(ppl_str) for ppl_str in all_ppl_str]

all_proportion=[]
for idx in range(len(all_ppl_attention)):
	attention_ppl = all_ppl_attention[idx]
	lstm_ppl      = all_ppl_lstm[idx]
	all_proportion.append(lstm_ppl/attention_ppl)

good_position = heapq.nlargest( extreme_number, range(len(all_proportion)), all_proportion.__getitem__)
bad_position  = heapq.nsmallest(extreme_number, range(len(all_proportion)), all_proportion.__getitem__)

with open(data_file , 'rb') as f_ori, open(log_file , 'wb') as f_log, open(good_file , 'wb') as f_good,open(bad_file , 'wb') as f_bad:
	all_data = f_ori.read().splitlines()
	f_log.write('good performance\n')
	for good_idx in good_position :
		f_log.write('lstm: ')
		f_log.write(str(all_ppl_lstm[good_idx]))
		f_log.write('  attention: ')
		f_log.write(str(all_ppl_attention[good_idx]))
		f_log.write('\n')

		f_good.write(all_data[good_idx])
		f_good.write('\n')

	f_log.write('bad performance\n')
	for bad_idx in bad_position :
		f_log.write('lstm: ')
		f_log.write(str(all_ppl_lstm[bad_idx]))
		f_log.write('  attention: ')
		f_log.write(str(all_ppl_attention[bad_idx]))
		f_log.write('\n')

		f_bad.write(all_data[bad_idx])
		f_bad.write('\n')





