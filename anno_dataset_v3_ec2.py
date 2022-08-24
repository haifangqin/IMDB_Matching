import pandas as pd
import numpy as np
import math
import sys
from collections import defaultdict
import Levenshtein
import nltk
import json
import os
from timeit import default_timer as timer
from multiprocessing import Pool
import pickle
import psutil



class CmsVideo():
    __slots__ = ('contentid','title','year','duration','cleaned_title','actors','directors','matched','match_director', 'match_actors', 'exact_match_id','final_match_id',
    'fuzzy_match_id','word_list')
    def __init__(self, contentid):
        self.contentid = contentid
        self.title = []
        self.year = []
        self.duration = []
        self.cleaned_title = []
        self.actors = []
        self.directors = []
        self.matched = False
        self.match_director = False
        self.match_actors = False
        self.exact_match_id = []
        self.final_match_id = []
        self.fuzzy_match_id = []
        self.word_list = []

    def get_lower_title(self):
        self.title = [_title.lower() if is_title_valid(_title) else _title for _title in self.title]

    def get_word_list(self, stopwords):
        for _title in self.title:
            if not is_title_valid(_title):
                continue
            self.word_list.extend(_title.split(' '))
        self.word_list.extend(self.cleaned_title)
        #self.word_list = [word.strip() for _word in self.word_list]
        self.word_list = [''.join(filter(str.isalnum, _word.strip())) if is_title_valid(_word) else _word.strip() for _word in self.word_list]
        temp_list = set(self.word_list) - stopwords
        if len(temp_list) > 0:
            self.word_list = temp_list
        else:
            self.word_list = set(self.word_list)
    
    def get_director_names(self):
        for name in self.directors:
            if is_title_valid(name):
                names = name[1:-1].split(',')
                names = [ _word[1:-1].lower() for _word in names]
                self.directors = names
            else:
                self.directors = []
        return

    def get_actors_names(self):
        for name in self.actors:
            if is_title_valid(name):
                names = name[1:-1].split(',')
                names = [ _word[1:-1].lower() for _word in names]
                self.actors = names
            else:
                self.actors = []
        return

    def get_cleaned_title(self):
        self.cleaned_title = [''.join(filter(str.isalnum, _title)) if is_title_valid(_title) else _title for _title in self.title]
    
    def print_cms(self):
        print('*'*20)
        print(self.contentid)
        print(self.title)
        print(self.cleaned_title)
        print(self.word_list)
        print(self.duration)
        print(self.actors)

class ImdbVideo():
    __slots__ = ('contentid','title','year','duration','cleaned_title','actors','directors','primary_title','original_title', 'word_list')
    def __init__(self, titleid):
        self.contentid = titleid
        self.title = []
        self.primary_title = []
        self.original_title = []
        self.year = []
        self.duration = []
        self.actors = []
        self.directors = []
        self.cleaned_title = []
        self.word_list = []

    def get_lower_title(self):
        self.title = [_title.lower() if is_title_valid(_title) else _title for _title in self.title]

    def get_cleaned_title(self):
        self.cleaned_title = [''.join(filter(str.isalnum, _title)) if is_title_valid(_title) else _title for _title in self.title]

    def get_word_list(self, stopwords):
        for _title in self.title:
            if not is_title_valid(_title):
                continue
            self.word_list.extend(_title.split(' '))
        for _title in self.cleaned_title:
            if not is_title_valid(_title):
                continue
            self.word_list.extend(_title)
        self.word_list = [''.join(filter(str.isalnum, _word.strip())) if is_title_valid(_word) else _word.strip() for _word in self.word_list]
        temp_list = set(self.word_list) - stopwords
        if len(temp_list) > 0:
            self.word_list = temp_list
        else:
            self.word_list = set(self.word_list)

    def get_directors(self, nconsts, df):
        for nconst in nconsts:
            if nconst in df:
                self.directors.extend(df[nconst])
        list(set(self.directors))
    
    def get_actors(self, nconsts, df):
        for nconst in nconsts:
            if nconst in df:
                self.actors.extend(df[nconst])
        list(set(self.actors))

    def print_imdb(self):
        print('-'*20)
        print(self.contentid)
        print(self.title)
        print(self.cleaned_title)
        print(self.word_list)
        print(self.duration)
        print(self.actors)

def get_stopwords():
    #nltk.download("stopwords")
    #stopwords = nltk.corpus.stopwords.words("english")
    #stopwords.append('')
    stopwords = ['on', 'at', 'in', 'of', 'the', '', 'trailer']
    print("get {} stop words!".format(len(stopwords)))
    return stopwords

def read_imdb_data():
    """
    read imdb data from csv file
    return: title_akas, title_basics
    """

    title_akas = pd.read_csv('../imdb_data/title.akas.tsv', sep='\t')
    title_basics = pd.read_csv('../imdb_data/title.basics.tsv', sep='\t')
    title_principals = pd.read_csv('../imdb_data/title.principals.tsv', sep='\t')
    title_crew = pd.read_csv('../imdb_data/title.crew.tsv', sep='\t')
    name_basics = pd.read_csv('../imdb_data/name.basics.tsv', sep='\t')

    #title_akas = pd.read_csv('../imdb_small_data/title.akas.small.tsv', sep='\t')
    #title_basics = pd.read_csv('../imdb_small_data/title.basics.small.tsv', sep='\t')
    #title_principals = pd.read_csv('../imdb_small_data/title.principals.small.tsv', sep='\t')
    #title_crew = pd.read_csv('../imdb_small_data/title.crew.small.tsv', sep='\t')
    #name_basics = pd.read_csv('../imdb_small_data/name.basics.small.tsv', sep='\t')
    return title_akas,  title_basics, title_principals, title_crew, name_basics


def pickle_dump_object(out_file_name='', dump_objs=None, fout=None):
    """
    docstring for pickle_dump_object
    """
    if dump_objs is None:
        dump_objs = []
    if fout is None:
        fout = open(out_file_name, 'wb')
    if isinstance(dump_objs, list):
        for obj in dump_objs:
            pickle.dump(obj, fout, -1)
    elif isinstance(dump_objs, dict):
        for obj in dump_objs.values():
            pickle.dump(obj, fout, -1)
    else:
        pickle.dump(dump_objs, fout, -1)

def pickle_load_object(in_file_name):
    """docstring for pickle_object"""
    with open(in_file_name, 'rb') as fin:
        while 1:
            try:
                obj = pickle.load(fin)
            except pickle.PickleError as error:
                pass
            except EOFError:
                break
            else:
                yield obj
def get_imdb_obj_list():

    if os.path.isfile('imdb_obj_list.pkl'):
        imdb_obj_list = []
        for imdb_obj in pickle_load_object('imdb_obj_list.pkl'):
            imdb_obj_list.append(imdb_obj)
        return imdb_obj_list
    else:
        return None


def get_dict_from_imdb():

    title_akas, title_basics, title_principals, tile_crew, name_basics = read_imdb_data()

    title_akas = title_akas[title_akas.title != '']
    title_akas.reset_index(inplace=True)

    # delete the episode type
    title_basics = title_basics[title_basics.titleType != 'tvEpisode']
    title_basics = title_basics[title_basics.titleType != 'tvSeries']
    title_basics = title_basics[title_basics.titleType != 'tvMiniSeries']
    title_basics.reset_index(inplace=True)

    if os.path.isfile('id_to_akas_title.json'):
        print("read a exist akas file!")
        with open('id_to_akas_title.json', 'r') as fp:
            id_to_akas_title = json.load(fp)
    else:
        id_to_akas_title = title_akas.groupby('titleId')['title'].apply(list).to_dict()
        with open('id_to_akas_title.json', 'w') as fp:
            json.dump(id_to_akas_title, fp)

    if os.path.isfile('id_to_basics_ptitle.json'):
        print("read a exist basics_ptitle file!")
        with open('id_to_basics_ptitle.json', 'r') as fp:
            id_to_basics_ptitle = json.load(fp)
    else:
        id_to_basics_ptitle = title_basics.groupby('tconst')['primaryTitle'].apply(list).to_dict()
        with open('id_to_basics_ptitle.json', 'w') as fp:
            json.dump(id_to_basics_ptitle, fp)

    if os.path.isfile('id_to_basics_otitle.json'):
        print("read a exist basics_otitle file!")
        with open('id_to_basics_otitle.json', 'r') as fp:
            id_to_basics_otitle = json.load(fp)
    else:
        id_to_basics_otitle = title_basics.groupby('tconst')['originalTitle'].apply(list).to_dict()
        with open('id_to_basics_otitle.json', 'w') as fp:
            json.dump(id_to_basics_otitle, fp)

    if os.path.isfile('id_to_basics_titleType.json'):
        print("read a exist id_to_basics_titleType file!")
        with open('id_to_basics_titleType.json', 'r') as fp:
            id_to_basics_titleType = json.load(fp)
    else:
        id_to_basics_titleType = title_basics.groupby('tconst')['titleType'].apply(list).to_dict()
        with open('id_to_basics_titleType.json', 'w') as fp:
            json.dump(id_to_basics_titleType, fp)

    if os.path.isfile('id_to_basics_year.json'):
        print("read a exist id_to_basics_year file!")
        with open('id_to_basics_year.json', 'r') as fp:
            id_to_basics_year = json.load(fp)
    else:
        id_to_basics_year = title_basics.groupby('tconst')['startYear'].apply(list).to_dict()
        with open('id_to_basics_year.json', 'w') as fp:
            json.dump(id_to_basics_year, fp)

    if os.path.isfile('id_to_basics_runtime.json'):
        print("read a exist id_to_basics_runtime file!")
        with open('id_to_basics_runtime.json', 'r') as fp:
            id_to_basics_runtime = json.load(fp)
    else:
        id_to_basics_runtime = title_basics.groupby('tconst')['runtimeMinutes'].apply(list).to_dict()
        with open('id_to_basics_runtime.json', 'w') as fp:
            json.dump(id_to_basics_runtime, fp)

    if os.path.isfile('id_to_crew_ndirectors.json'):
        print("read a exist id_to_crew_ndirectors file!")
        with open('id_to_crew_ndirectors.json', 'r') as fp:
            id_to_crew_ndirectors = json.load(fp)
    else:
        id_to_crew_ndirectors = tile_crew.groupby('tconst')['directors'].apply(list).to_dict()
        with open('id_to_crew_ndirectors.json', 'w') as fp:
            json.dump(id_to_crew_ndirectors, fp)


    if os.path.isfile('id_to_principals_ndirectors.json'):
        print("read a exist id_to_principals_ndirectors file!")
        with open('id_to_principals_ndirectors.json', 'r') as fp:
            id_to_principals_ndirectors = json.load(fp)
    else:
        id_to_principals_ndirectors = title_principals[title_principals['category']  == 'director'].groupby('tconst')['nconst'].apply(list).to_dict()
        with open('id_to_principals_ndirectors.json', 'w') as fp:
            json.dump(id_to_principals_ndirectors, fp)

    if os.path.isfile('id_to_principals_nactors.json'):
        print("read a exist id_to_principals_nactors file!")
        with open('id_to_principals_nactors.json', 'r') as fp:
            id_to_principals_nactors = json.load(fp)
    else:
        id_to_principals_nactors = title_principals[(title_principals['category']  == 'actor') | (title_principals['category']  == 'actress')].groupby('tconst')['nconst'].apply(list).to_dict()
        with open('id_to_principals_nactors.json', 'w') as fp:
            json.dump(id_to_principals_nactors, fp)

    if os.path.isfile('id_to_name.json'):
        print("read a exist id_to_name file!")
        with open('id_to_name.json', 'r') as fp:
            id_to_name = json.load(fp)
    else:
        id_to_name = name_basics.groupby('nconst')['primaryName'].apply(list).to_dict()
        with open('id_to_name.json', 'w') as fp:
            json.dump(id_to_name, fp)

    titleid_set = list(set(title_akas.titleId.to_list()).union(set(title_basics.tconst.to_list())))

    return titleid_set, id_to_akas_title, id_to_basics_ptitle, id_to_basics_otitle, id_to_basics_year, id_to_basics_runtime, id_to_crew_ndirectors, id_to_principals_ndirectors, id_to_principals_nactors, id_to_name, id_to_basics_titleType

def is_title_valid(val):
    if not isinstance(val, str):
        print("{} is not a str type in title".format(val))
        return 0
    if val == '\\N' or val  == 'NA\\' or val  ==  '\\NA\\' or val  == 'nan':
        return 0
    return 1

def is_valid(val):
    if val == '\\N' or val  == 'NA\\' or val  ==  '\\NA\\' or val  == 'nan':
        return 0
    return 1



if __name__  == "__main__":

    # CMS file name, .csv
    test_set_name = sys.argv[1]
    # Results save path, .xlsx
    save_excel_name = sys.argv[2]
    # Leven distance threshold, 0.8/0.9
    dis_thresh = sys.argv[3]
    
    # read cms table for matching
    test_set = pd.read_csv(test_set_name,  sep=',', encoding='latin-1')
    print('1111')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    # read all the IMDB table and process to dictionaries
    titleid_set, id_to_akas_title, id_to_basics_ptitle, id_to_basics_otitle, id_to_basics_year, id_to_basics_runtime, id_to_crew_ndirectors, id_to_principals_ndirectors, id_to_principals_nactors, id_to_name, id_to_basics_titleType = get_dict_from_imdb()
    print('1111')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    # get stop words, e.g. on, at, in...
    stopwords = get_stopwords()
    stopwords = set(stopwords)

    # process IMDB table to imdb_obj_list, for each object, initialize the title, actors, directors, duration and so on.
    id_to_actors = {}
    id_to_directors = {}
    # if there are imdb obj list there, you can read it and use it directly
    imdb_obj_list = get_imdb_obj_list()
    start = timer()
    if imdb_obj_list == None:
        imdb_obj_list = []
        for i, title_id in enumerate(titleid_set):
            print("\r",  "progress imdb percentage:{0}%".format((round(i + 1) * 100 / len(titleid_set))),  end = "",  flush = True)
            temp_obj = ImdbVideo(title_id)
            temp_obj.title = [] if title_id not in id_to_akas_title else id_to_akas_title[title_id]
            if title_id in id_to_basics_otitle:
                temp_obj.title.extend(id_to_basics_otitle[title_id])
                temp_obj.title.extend(id_to_basics_ptitle[title_id])
                temp_obj.title = list(set(temp_obj.title))
                temp_obj.year = id_to_basics_year[title_id]
                temp_obj.duration = id_to_basics_runtime[title_id]
            temp_obj.get_lower_title()
            temp_obj.get_cleaned_title()
            # the word list will delete the stop words
            temp_obj.get_word_list(stopwords)
            if title_id in id_to_crew_ndirectors:
                temp_obj.get_directors(id_to_crew_ndirectors[title_id], id_to_name)
            if title_id in id_to_principals_ndirectors:
                temp_obj.get_directors(id_to_principals_ndirectors[title_id], id_to_name)
            if title_id in id_to_principals_nactors:
                temp_obj.get_actors(id_to_principals_nactors[title_id], id_to_name)
            #if not i % 100:
                #temp_obj.print_imdb()
            imdb_obj_list.append(temp_obj)
            id_to_actors[title_id] = ','.join(temp_obj.actors)
            id_to_directors[title_id] = ','.join(temp_obj.directors)
        # save the object list to pkl file in order to read next time.
        pickle_dump_object('imdb_obj_list.pkl', imdb_obj_list)
        with open('id_to_actors.json', 'w') as fp:
            json.dump(id_to_actors, fp)
        with open('id_to_directors.json', 'w') as fp:
            json.dump(id_to_directors, fp)
        print("write the imdb obj list done...")
    else:
        with open('id_to_actors.json', 'r') as fp:
            id_to_actors = json.load(fp)
        with open('id_to_directors.json', 'r') as fp:
            id_to_directors = json.load(fp)
    print('3333')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    end = timer()
    print(end - start)

    # process cms data to dictionaries.
    id_to_cms_title = test_set.groupby('contentid')['title'].apply(list).to_dict()
    id_to_cms_year = test_set.groupby('contentid')['year'].apply(list).to_dict()
    id_to_cms_duration = test_set.groupby('contentid')['duration'].apply(list).to_dict()
    id_to_cms_actors = test_set.groupby('contentid')['actors'].apply(list).to_dict()
    id_to_cms_directors = test_set.groupby('contentid')['directors'].apply(list).to_dict()

    content_set = list(set(test_set.contentid.to_list()))
    # Initialize all the cms objects with their attributes.
    cms_obj_list = []
    for i, content_id in enumerate(content_set):
        temp_obj = CmsVideo(content_id)
        temp_obj.title = id_to_cms_title[content_id]
        temp_obj.year = id_to_cms_year[content_id]
        temp_obj.duration = id_to_cms_duration[content_id]
        temp_obj.actors = id_to_cms_actors[content_id]
        temp_obj.directors = id_to_cms_directors[content_id]
        temp_obj.get_lower_title()
        temp_obj.get_cleaned_title()
        # the word list will delete the stop words
        temp_obj.get_word_list(stopwords) 
        temp_obj.get_director_names()
        temp_obj.get_actors_names()
        #if not i % 100:
            #temp_obj.print_cms()
        cms_obj_list.append(temp_obj)
    print('4444')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )


    # Start to match every content in cms_obj_list to imdb_obj_list
    for i, cms_obj in enumerate(cms_obj_list):
        print("\r",  "match imdb percentage:{0}%".format((round(i + 1) * 100 / len(cms_obj_list))),  end = "",  flush = True)
        leven_dis = {}
        for imdb_obj in imdb_obj_list:
            # if not matched any word, we can skip this imdb obj directly. It means the leven_dis will be 0.0
            if not cms_obj.word_list & imdb_obj.word_list: 
                continue
            leven_dis[imdb_obj.contentid] = 0
            for cms_title in cms_obj.cleaned_title:
                if not is_title_valid(cms_title):
                    continue
                for imdb_title in imdb_obj.cleaned_title:
                    if not is_title_valid(imdb_title):
                        continue
                    if imdb_title  == cms_title:
                        cms_obj.exact_match_id.append(imdb_obj.contentid)
                        leven_dis[imdb_obj.contentid] = 1.0
                    else:
                        if len(imdb_title) > 5 and len(cms_title) > 5 and (imdb_title in cms_title or cms_title in imdb_title) and (Levenshtein.ratio(cms_title, imdb_title) > 0.5):
                            cms_obj.fuzzy_match_id.append(imdb_obj.contentid)
                        leven_dis[imdb_obj.contentid] = max(leven_dis[imdb_obj.contentid], Levenshtein.ratio(cms_title, imdb_title))
        # find match id list, we prior choose the exact match ids
        if len(cms_obj.exact_match_id) > 0:
            cms_obj.matched = True
            match_ids = cms_obj.exact_match_id
        else:
            temp_leven_dis = {k:v for k, v in leven_dis.items() if float(v) >= float(dis_thresh)}
            temp_leven_dis = {k: v for k,  v in sorted(temp_leven_dis.items(), key=lambda item: item[1], reverse=True)} # the higher the better
            num_save = min(10, len(temp_leven_dis))
            temp_match_id = list(temp_leven_dis.keys())
            cms_obj.fuzzy_match_id.extend(temp_match_id[:num_save])
            match_ids = cms_obj.fuzzy_match_id

        # score and sort
        match_ids = list(set(match_ids))
        scores = {}
        for match_id in match_ids:
            temp_score = 1.0
            if match_id in id_to_basics_ptitle:
                #match year
                if is_valid(cms_obj.year[0]) & is_valid(id_to_basics_year[match_id][0]):
                    if cms_obj.year[0]  == id_to_basics_year[match_id][0]:
                        temp_score  += 5.0

                #match runtime
                time_thresh = cms_obj.duration[0] / 60 * 0.2
                if not is_valid(id_to_basics_runtime[match_id][0]):
                    temp_score += 0.0
                elif abs(cms_obj.duration[0] / 60 - float(id_to_basics_runtime[match_id][0])) < time_thresh:
                    temp_score += 3.0

                #match type
                if id_to_basics_titleType[match_id][0] == 'movie':
                    temp_score += 2.0
                if id_to_basics_titleType[match_id][0] == 'short':
                    temp_score += 1.0
                if id_to_basics_titleType[match_id][0] == 'tvSpecial':
                    temp_score += 1.0
            if match_id in id_to_directors:
                # match director
                temp_directors = id_to_directors[match_id].split(',')
                temp_directors = [word_.lower() for word_ in temp_directors]
                if set(temp_directors) & set(cms_obj.directors):
                    cms_obj.match_director = True
                    temp_score += 10.0
            if match_id in id_to_actors:
                # match actor
                temp_actors = id_to_actors[match_id].split(',')
                temp_actors = [word_.lower() for word_ in temp_actors]
                if set(temp_actors) & set(cms_obj.actors):
                    cms_obj.match_actors = True
                    temp_score += 10.0
            #title Levenshtein distance
            temp_score += 5.0 * leven_dis[match_id]
            scores[match_id] = temp_score
        # keep the threshold larger than threshold
        scores = {k: v for k,  v in sorted(scores.items(), key=lambda item: item[1], reverse=True)} # the higher the better
        num_save = 3
        num_save = min(num_save, len(scores))
        temp_match_id = list(scores.keys())
        cms_obj.final_match_id = temp_match_id[:num_save]
    
    print('5555')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    end_1 = timer()
    print(end_1 - end)
    print("write the cms obj list...")
    pickle_dump_object('cms_obj_list.pkl', cms_obj_list)
    # write to a format excel
    data_list = []
    
    for cms_obj in cms_obj_list:
        if len(cms_obj.final_match_id) > 0:
            for match_id in cms_obj.final_match_id:
                if match_id in id_to_basics_titleType and match_id in id_to_akas_title:
                    data_list.append(
                            {
                                'if_matched': True,
                                'exact_matched': cms_obj.matched,
                                'match_director': cms_obj.match_director,
                                'match_actors': cms_obj.match_actors,
                                'cms_title': cms_obj.title[0],
                                'imdb_multilanguage_title': ','.join(id_to_akas_title[match_id]),
                                'imdb_originaltitle': id_to_basics_otitle[match_id][0],
                                'imdb_primarytitle': id_to_basics_ptitle[match_id][0],
                                'cms_directors': cms_obj.directors,
                                'imdb_directors': id_to_directors[match_id],
                                'cms_year': cms_obj.year[0],
                                'imdb_year': id_to_basics_year[match_id][0],
                                'cms_duration': cms_obj.duration[0],
                                'imdb_duration': id_to_basics_runtime[match_id][0],
                                'cms_type': 'MOVIE',
                                'imdb_type': id_to_basics_titleType[match_id][0],
                                'cms_actors': cms_obj.actors,
                                'imdb_actors': id_to_actors[match_id],
                                'cms_contentid': cms_obj.contentid,
                                'imdb_titleid': match_id
                                }
                            )
                elif match_id in id_to_basics_titleType:
                    data_list.append(
                            {
                                'if_matched': True,
                                'exact_matched': cms_obj.matched,
                                'match_director': cms_obj.match_director,
                                'match_actors': cms_obj.match_actors,
                                'cms_title': cms_obj.title[0],
                                'imdb_multilanguage_title': '',
                                'imdb_originaltitle': id_to_basics_otitle[match_id][0],
                                'imdb_primarytitle': id_to_basics_ptitle[match_id][0],
                                'cms_directors': cms_obj.directors,
                                'imdb_directors': id_to_directors[match_id],
                                'cms_year': cms_obj.year[0],
                                'imdb_year': id_to_basics_year[match_id][0],
                                'cms_duration': cms_obj.duration[0],
                                'imdb_duration': id_to_basics_runtime[match_id][0],
                                'cms_type': 'MOVIE',
                                'imdb_type': id_to_basics_titleType[match_id][0],
                                'cms_actors': cms_obj.actors,
                                'imdb_actors': id_to_actors[match_id],
                                'cms_contentid': cms_obj.contentid,
                                'imdb_titleid': match_id
                                }
                            )
                else:
                    data_list.append(
                            {
                                'if_matched': True,
                                'exact_matched': cms_obj.matched,
                                'match_director': cms_obj.match_director,
                                'match_actors': cms_obj.match_actors,
                                'cms_title': cms_obj.title[0],
                                'imdb_multilanguage_title': ','.join(id_to_akas_title[match_id]),
                                'imdb_originaltitle': '',
                                'imdb_primarytitle': '',
                                'cms_directors': cms_obj.directors,
                                'imdb_directors': id_to_directors[match_id],
                                'cms_year': cms_obj.year[0],
                                'imdb_year': '',
                                'cms_duration': cms_obj.duration[0],
                                'imdb_duration': '',
                                'cms_type': 'MOVIE',
                                'imdb_type': '',
                                'cms_actors': cms_obj.actors,
                                'imdb_actors': id_to_actors[match_id],
                                'cms_contentid': cms_obj.contentid,
                                'imdb_titleid': match_id
                                }
                            )
        else:
            data_list.append(
                            {
                                'if_matched': False,
                                'exact_matched': cms_obj.matched,
                                'match_director': cms_obj.match_director,
                                'match_actors': cms_obj.match_actors,
                                'cms_title': cms_obj.title[0],
                                'imdb_multilanguage_title': '',
                                'imdb_originaltitle': '',
                                'imdb_primarytitle': '',
                                'cms_directors': cms_obj.directors,
                                'imdb_directors': '',
                                'cms_year': cms_obj.year[0],
                                'imdb_year': '',
                                'cms_duration': cms_obj.duration[0],
                                'imdb_duration': '',
                                'cms_type': 'MOVIE',
                                'imdb_type': '',
                                'cms_actors': cms_obj.actors,
                                'imdb_actors': '',
                                'cms_contentid': cms_obj.contentid,
                                'imdb_titleid': ''
                                }
                            )
    data = pd.DataFrame(data_list, columns = ['if_matched', 'exact_matched', 'match_director', 'match_actors', 'cms_title', 'imdb_originaltitle', 'imdb_primarytitle', 'cms_directors', 'imdb_directors',
        'cms_year', 'imdb_year', 'cms_duration', 'imdb_duration', 'cms_type', 'imdb_type', 'cms_actors', 'imdb_actors', 'cms_contentid', 'imdb_titleid', 'imdb_multilanguage_title'])
    data.to_excel(save_excel_name, index=False)
    print('6666')
    print ('the memory now is ：',psutil.Process(os.getpid()).memory_info().rss)
    print ('the memory now is：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    end_2 = timer()
    print(end_2 - end_1)
