select * from gamepositionrollup where (is_training_data + is_testing_data + is_validation_data) = 1;
select count(*) from gamepositionrollup where is_training_data = 1;
select count(*) from gamepositionrollup where is_testing_data = 1;
select count(*) from gamepositionrollup where is_validation_data = 1;