output test csv:    
1. model = load_model()   
2. voice = get_register_voice()
3. average_register_voice_feature = average(model.fit(voice)) # get_average_voice_vector_of_specific_person
4. test_voice = get_test_voice()
5. test_voice_feature = model.fit(test_voice)
6. result = get_top_5_nearest_voice_to_the_register_voice(average_register_voice_feature, test_voice_feature)
7. output_result_to_csv(result)
**************************************
model test:   
  * Leave a few directories out and train on the others
  * pick 3 voice files as the register voice
  * validate on the left voice
1. model = load_model()   
2. voice = get_register_voice()
3. test = get_test_voice()
4. get_vector()
5. distance(Anchor, vector, type)
6. auc