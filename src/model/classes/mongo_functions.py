from pymongo import MongoClient, InsertOne

import numpy as np
import hashlib
import json
from tqdm import tqdm
import time
import redis
import hiredis

from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.sqlite.models import GamePositions
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.cnn_bb_scorer import boardCnnEval

from chess_engine.src.model.classes.sqlite.dependencies import   fetch_all_game_positions_rollup,get_GamePositionRollup_row_size,create_rollup_table

class mongo_data_pipe():
    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        if "validation_collection_key" not in kwargs:
            self.validation_collection_key=s.validation_collection_key
        else:
            self.validation_collection_key = kwargs["validation_collection_key"]

        if "testing_collection_key" not in kwargs:
            self.testing_collection_key=s.testing_collection_key
        else:
            self.testing_collection_key = kwargs["testing_collection_key"]
        
        if "training_collection_key" not in kwargs:
            self.training_collection_key=s.training_collection_key
        else:
            self.training_collection_key = kwargs["training_collection_key"]

        if "main_collection_key" not in kwargs:
            self.main_collection_key=s.main_collection_key
        else:
            self.main_collection_key = kwargs["main_collection_key"]

        if "db_name" not in kwargs:
            self.db_name=s.db_name
        else:
            self.db_name = kwargs["db_name"]

        if "metadata_key" not in kwargs:
            self.metadata_key=s.metadata_key
        else:
            self.metadata_key = kwargs["metadata_key"]

        if "bitboards_key" not in kwargs:
            self.bitboards_key=s.bitboards_key
        else:
            self.bitboards_key = kwargs["bitboards_key"]
        
        if "results_key" not in kwargs:
            self.results_key=s.results_key
        else:
            self.results_key = kwargs["results_key"]



        if "mongo_url" not in kwargs:
            self.mongo_url=s.mongo_url
        else:
            self.mongo_url = kwargs["mongo_url"]



        
        if "nnTestSize" not in kwargs:
            self.nnTestSize=s.nnTestSize
        else:
            self.nnTestSize = kwargs["nnTestSize"]

        if "nnValidationSize" not in kwargs:
            self.nnValidationSize=s.nnValidationSize
        else:
            self.nnValidationSize = kwargs["nnValidationSize"]    

        if "BatchSize" not in kwargs:
            self.batch_size=s.BatchSize
        else:
            self.batch_size = kwargs["BatchSize"]  


        
        self.np_means_file = s.np_means_file
        self.np_stds_file = s.np_stds_file

        self.evaluator = boardCnnEval()    

        

    def create_client(self):
        client = MongoClient(self.mongo_url, maxPoolSize=100,w=1)
        return client

    def open_connections(self):
        self.client = self.create_client()
        self.db = self.client[self.db_name]

        self.main_collection = self.db[self.main_collection_key]
        self.test_collection = self.db[self.testing_collection_key]
        self.valid_collection = self.db[self.validation_collection_key]
        self.train_collection = self.db[self.training_collection_key]

        return 1
    
    def close_connections(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.main_collection = None
            self.test_collection = None
            self.valid_collection = None
            self.train_collection = None




    def collection_decider(self,hash_value: int, validation_min: int, test_min: int):
        if hash_value < validation_min:
            return self.validation_collection_key
        elif hash_value < test_min:
            return self.testing_collection_key
        else:
            return self.training_collection_key




    
    def shuffle_and_split_set(self ):
        i = 0
        collections_and_keys = [[self.testing_collection_key, self.test_collection],
                                [self.validation_collection_key, self.valid_collection], 
                                [self.training_collection_key, self.train_collection]]
        collected_docs = {self.testing_collection_key:[],
                        self.validation_collection_key:[],
                        self.training_collection_key:[]}
            
        validation_min = self.nnValidationSize * 100
        test_min = self.nnTestSize *100 + validation_min

        for doc in iteratingFunction(collection=self.main_collection):
        # for docs in mongo_document_generator(yield_size=yield_size):
            # for doc in docs:
                
            bin = get_hash_ring_value(id=doc['_id'])
            
            collection = self.collection_decider(hash_value=bin,
                                                 validation_min=validation_min,
                                                 test_min=test_min)
            
            collected_docs[collection].append(doc)
            
            i += 1
            if i >= self.batch_size:
                
                for key,collection_client in collections_and_keys:
                    if collected_docs[key] == []:
                        pass
                    else:
                        collection_client.insert_many(collected_docs[key])
                
                collected_docs = {self.testing_collection_key:[],
                    self.validation_collection_key:[],
                    self.training_collection_key:[]}
                i = 0
        return 1
    
    def shuffle_and_split_set_bulk(self):
        self.open_connections()
        validation_min = self.nnValidationSize * 100
        test_min = self.nnTestSize * 100 + validation_min
        collected_docs = {self.testing_collection_key: [], self.validation_collection_key: [], self.training_collection_key: []}
        collections_and_keys = [(self.testing_collection_key, self.test_collection), 
                                (self.validation_collection_key, self.valid_collection), 
                                (self.training_collection_key, self.train_collection)]

        i = 0
        for doc in iteratingFunction(collection=self.main_collection):
            bin = get_hash_ring_value(id=doc['_id'])
            collection = self.collection_decider(hash_value=bin, validation_min=validation_min, test_min=test_min)
            collected_docs[collection].append(doc)
            i += 1
            if i >= self.batch_size:
                self.bulk_insert(collected_docs, collections_and_keys)
                collected_docs = {self.testing_collection_key: [], self.validation_collection_key: [], self.training_collection_key: []}
                i = 0

        if any(collected_docs.values()):
            self.bulk_insert(collected_docs, collections_and_keys)
        return 1

    def bulk_insert(self, collected_docs, collections_and_keys):
        for key, collection_client in collections_and_keys:
            if collected_docs[key]:
                requests = [InsertOne(doc) for doc in collected_docs[key]]
                collection_client.bulk_write(requests)
            
    def initialize_data(self,batch_size: int = 512):
        self.delete_all_collection_documents()
        self.process_sqlite_boards_to_mongo(batch_size=batch_size)

        


        self.shuffle_and_split_set()

        train_mean, train_std = self.calc_mongo_train()
        
        np.save(self.np_means_file,train_mean)
        np.save(self.np_stds_file,train_std)

        return train_mean, train_std
    
    def delete_non_main_collections(self):
        self.valid_collection.delete_many({})
        self.test_collection.delete_many({})
        self.train_collection.delete_many({})
   

    def calc_mongo_train(self):
        collection_stats = self.db.command("collstats", self.training_collection_key)
        m = int(collection_stats['count']) 
        sample_doc = self.train_collection.find_one({})
        n = len(list(sample_doc['metadata']))
        mean = self.calc_mean(m=m,n=n)
        std = self.calc_std(m=m,n=n,mean=mean)
        return mean, std

    def calc_mean(self,m,n):
        mean = np.zeros((1, n))
        md_lists = []
        i  = 0
        total_docs = 0
        

            
        for doc in self.train_collection.find({}, {'metadata': 1,'_id': 0},batch_size = self.batch_size):
            md_lists.append(doc['metadata'])
            i += 1
            total_docs += 1
            if i >= self.batch_size:

                mean, md_lists = mean_aggregate(curr_md_list=md_lists, curr_mean=mean)
                i = 0
                
        if md_lists == []:
            mean, md_lists = mean_aggregate(curr_md_list=md_lists, curr_mean=mean)
            
        print(f"docs calcd: {total_docs}, m: {m}, n: {n}")
        mean = (1/m) * mean
        return mean
        

    def calc_std(self,m,n,mean):    
        std = np.zeros((1, n))
        md_lists = []
        i  = 0
        total_docs = 0
        
        for doc in self.train_collection.find({}, {'metadata': 1,'_id': 0},batch_size = self.batch_size):
            md_lists.append(doc['metadata'])
            i += 1
            total_docs += 1
            if i >= self.batch_size:
                std, md_lists = std_aggregate(curr_md_list = md_lists,curr_std = std,mean = mean)
                i = 0
                
        if md_lists == []:
            std, md_lists = std_aggregate(curr_md_list = md_lists,curr_std = std,mean = mean)
        print(f"docs calcd: {total_docs}, m: {m}, n: {n}")
        std = (1/m) * std
        std = np.sqrt(std)
        std = np.where(std == 0, 1e-10, std)
        return std











    def process_sqlite_boards_to_mongo(self,batch_size: int = 512):
        create_rollup_table(yield_size=batch_size)
        row_count = get_GamePositionRollup_row_size()
        batch = fetch_all_game_positions_rollup(yield_size=512)
        dataset = []  # List to accumulate serialized examples
        for game in tqdm(batch, total=row_count, desc="Processing Feature Data"):
            try:
                if game:
                    
                    document = self.game_to_doc_evaluation(game=game)

                    document['_id'] = get_hash_id(doc=document)
                    
                    dataset.append(InsertOne(document))



                    # Check if we've accumulated enough examples to write a batch
                    if len(dataset) >= batch_size:
                        
                        self.main_collection.bulk_write(dataset)
                        dataset = []  # Reset the list after writing
                else:
                    return 1
            except Exception as e:
                raise Exception(e)

    # def bulk_insert(self, collected_docs, collections_and_keys):
    #     for key, collection_client in collections_and_keys:
    #         if collected_docs[key]:
    #             requests = [InsertOne(doc) for doc in collected_docs[key]]
    #             collection_client.bulk_write(requests)
                
    def game_to_doc_evaluation(self,game):

        board_scores = self.evaluator.get_game_scores(game=game)

        document = {
            "metadata": board_scores['metadata'],  # Ensure this does not contain large integers
            "positions_data": convert_large_ints(board_scores['positions_data']),
            "game_results": board_scores['game_results']
        }
        return(document)
    

    def check_collection_sizes(self):
        collection_stats = self.db.command("collstats", self.testing_collection_key)
        print(f"Document Count: {collection_stats['count']}")
        collection_stats = self.db.command("collstats", self.validation_collection_key)
        print(f"Document Count: {collection_stats['count']}")
        collection_stats = self.db.command("collstats", self.training_collection_key)
        print(f"Document Count: {collection_stats['count']}")

    def delete_all_collection_documents(self):
        self.main_collection.delete_many({})
        self.valid_collection.delete_many({})
        self.test_collection.delete_many({})
        self.train_collection.delete_many({})
        return 1



def std_aggregate(curr_md_list,curr_std,mean):
    curr_md_list = np.array(curr_md_list)
    curr_md_list = (curr_md_list-mean) ** 2
    curr_md_list = np.sum(curr_md_list,axis = 0,keepdims=True)
    curr_std = curr_std + curr_md_list
    curr_md_list = []
    return curr_std, curr_md_list

def mean_aggregate(curr_md_list, curr_mean):
    curr_md_list = np.array(curr_md_list)
    curr_md_list = np.sum(curr_md_list,axis = 0,keepdims=True)
    curr_mean = curr_mean + curr_md_list
    curr_md_list = []
    return curr_mean, curr_md_list

def bitboard_to_matrix(bitboard):
    return np.array([(bitboard >> shift) & 1 for shift in range(64)]).reshape(8, 8)

def create_cnn_input(bitboards):
    layers = []
    for bb in bitboards:  # Ensure consistent order
        matrix = bitboard_to_matrix(int(bb))
        # print(matrix)
        layers.append(matrix)
    return np.stack(layers)

def get_hash_ring_value(id, bins=100):
    hash_value = int(id, 16) % bins
    return hash_value

def iteratingFunction(collection,batch_size: int = 1000):
    try:

        batch = collection.find({}, batch_size = batch_size)

        for doc in batch:
            yield doc

    except Exception as e:
        print("Error occured!: ",e)

def convert_large_ints(data):
    # Convert any large integers to strings to prevent OverflowError

    for i in range(0,len(data)):
        data[i] = str(data[i])
    return data

def get_hash_id(doc):
    """ Generate a hash value for the item ID and scale it to the range 0 to bins-1 """
    dict_string = json.dumps(doc, sort_keys=True)
    hash_object = hashlib.sha256(dict_string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

def iteratingFunctionScaled(collection,means,stds,batch_size: int = 1000):
    try:

        batch = collection.find({}, {'_id': 0}, batch_size = batch_size)

        for doc in batch:
            
            doc['positions_data'] = create_cnn_input(doc['positions_data'])
            doc['metadata'] = (doc['metadata'] - means) / stds
            doc['game_results'] = np.array(doc['game_results'])
            yield doc

    except Exception as e:
        print("Error occured!: ",e)
