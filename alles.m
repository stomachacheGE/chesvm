%esvm_train(feature, calibration,algorithm, hard_negative)
esvm_train('hog',false,'svm',true);
esvm_train('hog',false,'esvm',false);
esvm_train('hog',false,'esvm',true);
esvm_train('hog',true,'esvm',false);
esvm_train('hog',true,'esvm',true);

esvm_train('cnn',false,'svm',true);
esvm_train('cnn',false,'esvm',false);

esvm_train('cnn',true,'esvm',false);
esvm_train('cnn',true,'esvm',true);
esvm_train('cnn',false,'esvm',true);