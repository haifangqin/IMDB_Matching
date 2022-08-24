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
import datetime
import time

class CmsShow():
    def __init__(self, titleid):
        self.contentid = titleid #showid
        self.title = []#episode, show, season
        self.matched = False
        self.match_actors = False
        self.match_director = False

        self.startyear = []
        self.endyear = []

        self.actors = [] # show
        self.directors = [] # show
        self.cleaned_title = []
        self.word_list = []

        self.exact_match_id = []
        self.fuzzy_match_id = []
        self.final_match_id = []
    
    def get_lower_title(self):
        self.title = [_title.lower() if is_title_valid(_title) else _title for _title in self.title]
        self.title = list(set(self.title))


    def get_cleaned_title(self):
        self.cleaned_title = [''.join(filter(str.isalnum, _title)) if is_title_valid(_title) else _title for _title in self.title]
        self.cleaned_title = list(set(self.cleaned_title))

    def get_word_list(self, stopwords):
        for _title in self.title:
            if not is_title_valid(_title):
                continue
            self.word_list.extend(_title.split(' '))
        for _title in self.cleaned_title:
            if not is_title_valid(_title):
                continue
            self.word_list.append(_title)

        self.word_list = [''.join(filter(str.isalnum, _word.strip())) if is_title_valid(_word) else _word for _word in self.word_list]
        
        temp_list = set(self.word_list) - stopwords
        if len(temp_list) > 0:
            self.word_list = temp_list
        else:
            self.word_list = set(self.word_list)
    
    def print_cms(self):
        print('-'*20)
        print(self.contentid)
        print(self.title)
        print(self.startyear)
        print(self.endyear)
        print(self.cleaned_title)
        print(self.word_list)
        print(self.actors)
        print(self.directors)


class ImdbShow():
    def __init__(self, titleid):
        self.contentid = titleid
        self.title = [] # parent title e.g I viaggi di DonnAvventura
        
        self.startyear = []
        self.endyear = []
        
        self.duration = {} # for every episode
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
            self.word_list.extend(self.cleaned_title)
        self.word_list = [''.join(filter(str.isalnum, _word.strip())) if is_title_valid(_word) else _word for _word in self.word_list]
        temp_list = set(self.word_list) - stopwords
        if len(temp_list) > 0:
            self.word_list = temp_list
        else:
            self.word_list = set(self.word_list)

    def get_directors(self, nconsts, df):
        for nconst in nconsts:
            if nconst in df:
                self.directors.extend(df[nconst])
        self.directors = list(set(self.directors))
    def get_actors(self, nconsts, df):
        for nconst in nconsts:
            if nconst in df:
                self.actors.extend(df[nconst])
        self.actors = list(set(self.actors))

    def print_imdb(self):
        print('-'*20)
        print(self.contentid)
        print(self.title)
        print(self.cleaned_title)
        print(self.word_list)
        print(self.duration)
        print(self.actors)
        print(self.directors)

def get_stopwords():
    #nltk.download("stopwords")
    #stopwords = nltk.corpus.stopwords.words("english")
    #stopwords.append('')
    stopwords = ['on','at','in','of','the','']
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
    title_episode = pd.read_csv('../imdb_data/title.episode.tsv',sep='\t')

    return title_akas,  title_basics, title_principals, title_crew, name_basics, title_episode


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

def pickle_load_object_index(in_file_name='', key='id'):
    """TODO: Docstring for pickle_load_object_index.
    :in_file_name: TODO
    :key: TODO
    :returns: TODO
    """
    if not in_file_name:
        return
    obj_index = {}
    for obj in pickle_load_object(in_file_name):
        key_id = getattr(obj, key) if hasattr(obj, key) else len(obj_index)
        obj_index[key_id] = obj
    return obj_index

def get_imdb_obj_list():

    if os.path.isfile('imdb_obj_list.pkl'):
        imdb_obj_list = []
        for imdb_obj in pickle_load_object('imdb_obj_list.pkl'):
            imdb_obj_list.append(imdb_obj)
        return imdb_obj_list
    else:
        return None

def get_imdb_episode_obj_list():
    if os.path.isfile('imdb_episode_obj_list.pkl'):
        imdb_obj_list = []
        for imdb_obj in pickle_load_object('imdb_episode_obj_list.pkl'):
            imdb_obj_list.append(imdb_obj)
        print('read imdb_episode obj list {}...'.format(len(imdb_obj_list)))
        return imdb_obj_list
    else:
        return None

def get_imdb_episode_obj_dict():
    if os.path.isfile('imdb_episode_obj_dict.pkl'):
        imdb_obj_dict = pickle_load_object_index('imdb_episode_obj_list.pkl', key='contentid')
        return imdb_obj_dict
    else:
        return None

def get_cms_episode_obj_list():
    if os.path.isfile('cms_episode_obj_list.pkl'):
        imdb_obj_list = []
        for imdb_obj in pickle_load_object('cms_episode_obj_list.pkl'):
            imdb_obj_list.append(imdb_obj)
        print('read cms_episode obj list {}...'.format(len(imdb_obj_list)))
        return imdb_obj_list
    else:
        return None

def get_cms_show_obj_list():
    if os.path.isfile('cms_show_obj_list.pkl'):
        imdb_obj_list = []
        for imdb_obj in pickle_load_object('cms_show_obj_list.pkl'):
            imdb_obj_list.append(imdb_obj)
        print('read cms_showobj list {}...'.format(len(imdb_obj_list)))
        return imdb_obj_list
    else:
        return None

def get_cms_id_to_imdb():
    if os.path.isfile('cms_id_to_imdb.json'):
        with open('cms_id_to_imdb.json', 'r') as fp:
            cms_id_to_imdb = json.load(fp)
        print('read the cms_id_to_imdb file {}...',format(len(cms_id_to_imdb)))
        return cms_id_to_imdb
    else:
        return None

def get_dict_from_imdb():

    title_akas, title_basics, title_principals, tile_crew, name_basics, title_episode = read_imdb_data()

    title_akas = title_akas[title_akas.title != '']
    title_akas.reset_index(inplace=True)

    # delete the movie type

    title_basics = title_basics[title_basics.titleType != 'short']
    title_basics = title_basics[title_basics.titleType != 'movie']
    title_basics = title_basics[title_basics.titleType != 'video']
    title_basics = title_basics[title_basics.titleType != 'tvSpecial']
    title_basics = title_basics[title_basics.titleType != 'videoGame']
    title_basics = title_basics[title_basics.titleType != 'tvShort']
    title_basics.reset_index(inplace=True)

    print("title basics has length: {}".format(title_basics.shape[0]))
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
        print("read a exist basics_titleType file!")
        with open('id_to_basics_titleType.json', 'r') as fp:
            id_to_basics_titleType = json.load(fp)
    else:
        id_to_basics_titleType = title_basics.groupby('tconst')['titleType'].apply(list).to_dict()
        with open('id_to_basics_titleType.json', 'w') as fp:
            json.dump(id_to_basics_titleType, fp)

    if os.path.isfile('id_to_basics_year.json'):
        print("read a exist basics_year file!")
        with open('id_to_basics_year.json', 'r') as fp:
            id_to_basics_year = json.load(fp)
    else:
        id_to_basics_year = title_basics.groupby('tconst')['startYear'].apply(list).to_dict()
        with open('id_to_basics_year.json', 'w') as fp:
            json.dump(id_to_basics_year, fp)

    if os.path.isfile('id_to_basics_endyear.json'):
        print("read a exist basics_endyear file!")
        with open('id_to_basics_endyear.json', 'r') as fp:
            id_to_basics_endyear = json.load(fp)
    else:
        id_to_basics_endyear = title_basics.groupby('tconst')['endYear'].apply(list).to_dict()
        with open('id_to_basics_endyear.json', 'w') as fp:
            json.dump(id_to_basics_endyear, fp)

    if os.path.isfile('id_to_basics_runtime.json'):
        print("read a exist basics_runtime file!")
        with open('id_to_basics_runtime.json', 'r') as fp:
            id_to_basics_runtime = json.load(fp)
    else:
        id_to_basics_runtime = title_basics.groupby('tconst')['runtimeMinutes'].apply(list).to_dict()
        with open('id_to_basics_runtime.json', 'w') as fp:
            json.dump(id_to_basics_runtime, fp)

    if os.path.isfile('id_to_crew_ndirectors.json'):
        print("read a exist crew_directors file!")
        with open('id_to_crew_ndirectors.json', 'r') as fp:
            id_to_crew_ndirectors = json.load(fp)
    else:
        id_to_crew_ndirectors = tile_crew.groupby('tconst')['directors'].apply(list).to_dict()
        with open('id_to_crew_ndirectors.json', 'w') as fp:
            json.dump(id_to_crew_ndirectors, fp)


    if os.path.isfile('id_to_principals_ndirectors.json'):
        print("read a exist principals_directors file!")
        with open('id_to_principals_ndirectors.json', 'r') as fp:
            id_to_principals_ndirectors = json.load(fp)
    else:
        id_to_principals_ndirectors = title_principals[title_principals['category']  == 'director'].groupby('tconst')['nconst'].apply(list).to_dict()
        with open('id_to_principals_ndirectors.json', 'w') as fp:
            json.dump(id_to_principals_ndirectors, fp)

    if os.path.isfile('id_to_principals_nactors.json'):
        print("read a exist principals_actors file!")
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
    
    if os.path.isfile('id_to_episode_parent.json'):
        print("read a exist id_to_episode_parent file!")
        with open('id_to_episode_parent.json', 'r') as fp:
            id_to_episode_parent = json.load(fp)
    else:
        id_to_episode_parent = title_episode.groupby('tconst')['parentTconst'].apply(list).to_dict()
        with open('id_to_episode_parent.json', 'w') as fp:
            json.dump(id_to_episode_parent, fp)

    if os.path.isfile('id_to_seasonno.json'):
        print("read a exist id_to_seasonno file!")
        with open('id_to_seasonno.json', 'r') as fp:
            id_to_seasonno = json.load(fp)
    else:
        id_to_seasonno = title_episode.groupby('tconst')['seasonNumber'].apply(list).to_dict()
        with open('id_to_seasonno.json', 'w') as fp:
            json.dump(id_to_seasonno, fp)

    if os.path.isfile('id_to_episodeno.json'):
        print("read a exist id_to_episodeno file!")
        with open('id_to_episodeno.json', 'r') as fp:
            id_to_episodeno = json.load(fp)
    else:
        id_to_episodeno = title_episode.groupby('tconst')['episodeNumber'].apply(list).to_dict()
        with open('id_to_episodeno.json', 'w') as fp:
            json.dump(id_to_episodeno, fp)

    titleid_set = list(set(title_basics.tconst.to_list()))
    
    return titleid_set, id_to_akas_title, id_to_basics_ptitle, id_to_basics_otitle, id_to_basics_year, id_to_basics_endyear, id_to_basics_runtime, id_to_crew_ndirectors, id_to_principals_ndirectors, id_to_principals_nactors, id_to_name, id_to_basics_titleType, id_to_episode_parent, id_to_seasonno, id_to_episodeno

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

def read_cms_data():
    test_show = pd.read_csv('../cms_data/in_cms_tvshow_2021_06_16.csv',sep=',')
    return test_show

def read_cms_episode_data():
    test_episode = pd.read_csv('../cms_data/in_cms_episode_update_s3.csv',sep=',')
    return test_episode

if __name__  == "__main__":
    # CMS file name, .csv
    test_set_name = sys.argv[1]
    # Results save path, .xlsx
    save_excel_name = sys.argv[2]
    # Leven distance threshold, 0.8/0.9
    dis_thresh = sys.argv[3]

    # read cms table for matching
    test_set = pd.read_csv(test_set_name,  sep=',') # episode
    test_show = read_cms_data()

    # get stop words, e.g. on, at, in...
    stopwords = get_stopwords()
    stopwords = set(stopwords)

    # read all the IMDB table and process to dictionaries
    titleid_set, id_to_akas_title, id_to_basics_ptitle, id_to_basics_otitle, id_to_basics_year, id_to_basics_endyear, id_to_basics_runtime, id_to_crew_ndirectors, id_to_principals_ndirectors, id_to_principals_nactors, id_to_name, id_to_basics_titleType, id_to_episode_parent, id_to_seasonno, id_to_episodeno = get_dict_from_imdb()


    # process IMDB table to imdb_obj_list, for each object, initialize the title, actors, directors, duration and so on.
    id_to_actors = {}
    id_to_directors = {}
    id_to_season_episode = {}
    # if there are imdb obj list there, you can read it and use it directly
    imdb_obj_list = get_imdb_episode_obj_list()
    start = timer()
    if imdb_obj_list == None:
        imdb_obj_list = {}
        for i, showid in enumerate(titleid_set):
            print("\r",  "progress imdb percentage:{0}%".format((round(i + 1) * 100 / len(titleid_set))),  end = "",  flush = True)
            if not showid in imdb_obj_list:
                if showid not in id_to_basics_otitle and showid not in id_to_akas_title:
                    continue
                temp_obj = ImdbShow(showid) # showid
                temp_obj.title = [] if showid not in id_to_akas_title else id_to_akas_title[showid]
                if showid in id_to_basics_otitle:
                    temp_obj.title.extend(id_to_basics_otitle[showid])
                    temp_obj.title.extend(id_to_basics_ptitle[showid])
                    temp_obj.title = list(set(temp_obj.title))
                    temp_obj.startyear = id_to_basics_year[showid]
                    temp_obj.endyear = id_to_basics_endyear[showid]
                    temp_obj.duration = id_to_basics_runtime[showid]
                temp_obj.get_lower_title()
                temp_obj.get_cleaned_title()
                # the word list will delete the stop words
                temp_obj.get_word_list(stopwords)
                if showid in id_to_crew_ndirectors:
                    temp_obj.get_directors(id_to_crew_ndirectors[showid], id_to_name)
                if showid in id_to_principals_ndirectors:
                    temp_obj.get_directors(id_to_principals_ndirectors[showid], id_to_name)
                if showid in id_to_principals_nactors:
                    temp_obj.get_actors(id_to_principals_nactors[showid], id_to_name)
                imdb_obj_list[showid] = temp_obj
                id_to_actors[showid] = ','.join(temp_obj.actors)
                id_to_directors[showid] = ','.join(temp_obj.directors)
            else: # get a exist show
                temp_obj = imdb_obj_list[showid]
        imdb_obj_dict = imdb_obj_list
        imdb_obj_list = list(imdb_obj_list.values())
        # save the object list to pkl file in order to read next time.
        pickle_dump_object('imdb_episode_obj_list.pkl', imdb_obj_list)
        pickle_dump_object('imdb_episode_obj_dict.pkl', imdb_obj_dict)
        with open('id_to_actors.json', 'w') as fp:
            json.dump(id_to_actors, fp)
        with open('id_to_directors.json', 'w') as fp:
            json.dump(id_to_directors, fp)
    else:
        imdb_obj_dict = get_imdb_episode_obj_dict()
        with open('id_to_actors.json', 'r') as fp:
            id_to_actors = json.load(fp)
        with open('id_to_directors.json', 'r') as fp:
            id_to_directors = json.load(fp)
    
    end = timer()
    print(end - start)
    
    # process cms episode data
    #id_to_cms_showid = test_set.groupby('contentid')['showid'].apply(list).to_dict()
    #showid_to_episode_id = test_set.groupby('showid')['contentid'].apply(list).to_dict()
    id_to_cms_showshorttitle = test_set.groupby('showid')['showshorttitle'].apply(set).to_dict()
    id_to_cms_showname = test_set.groupby('showid')['showname'].apply(set).to_dict()
    id_to_episode_actors = test_set.groupby('showid')['actors'].apply(set).to_dict()
    id_to_episode_directors = test_set.groupby('showid')['directors'].apply(set).to_dict()

    #process the actors
    for key, value in id_to_episode_actors.items():
        temp_list = []
        for names in value:
            if isinstance(names, float):
                continue
            actors = names[1:-1].split(',')
            for actor in actors:
                temp_list.append(actor[1:-1].lower())
        id_to_episode_actors[key] = list(set(temp_list))
    #process the directors
    for key, value in id_to_episode_directors.items():
        temp_list = []
        for names in value:
            if isinstance(names, float):
                continue
            actors = names[1:-1].split(',')
            for actor in actors:
                temp_list.append(actor[1:-1].lower())
        id_to_episode_directors[key] = list(set(temp_list))

    #id_to_show_id = test_show.groupby('showid')['contentid'].apply(list).to_dict()
    id_to_show_startdt = test_show.groupby('id')['startdt'].apply(list).to_dict()
    id_to_show_enddt = test_show.groupby('id')['enddt'].apply(list).to_dict()
    id_to_show_title = test_show.groupby('id')['title'].apply(list).to_dict()
    
    content_set = list(set(test_show.id.to_list()))
    # process cms data to dictionaries.
    # Initialize all the cms objects with their attributes.
    cms_show_obj_list = get_cms_show_obj_list()
    if cms_show_obj_list == None:
        cms_show_obj_list = {}
        for i, showid in enumerate(content_set):
            print("\r",  "progress cms episode percentage:{0}%".format((round(i + 1) * 100 / len(content_set))),  end = "",  flush = True)
            if showid not in cms_show_obj_list:
                temp_obj = CmsShow(showid)
                temp_obj.title = id_to_show_title[showid]
                
                temp_obj.startyear = time.gmtime(id_to_show_startdt[showid][0]).tm_year
                temp_obj.endyear = time.gmtime(id_to_show_enddt[showid][0]).tm_year
                
                if showid in id_to_cms_showshorttitle:
                    temp_obj.title.extend(id_to_cms_showshorttitle[showid])
                if showid in id_to_cms_showname:
                    temp_obj.title.extend(id_to_cms_showname[showid])
                if showid in id_to_episode_actors:
                    temp_obj.actors.extend(id_to_episode_actors[showid])
                if  showid in id_to_episode_directors:
                    temp_obj.directors.extend(id_to_episode_directors[showid])
                
                cms_show_obj_list[showid] = temp_obj
                temp_obj.get_lower_title()
                temp_obj.get_cleaned_title()
                temp_obj.get_word_list(stopwords)
                
            else:
                temp_obj = cms_show_obj_list[showid]
            temp_obj.print_cms()

        cms_show_obj_list = list(cms_show_obj_list.values())
        pickle_dump_object('cms_show_obj_list.pkl', cms_show_obj_list)

    else:
        pass
    
    # Start to match every content in cms_obj_list to imdb_obj_list
    print('match imdb and cms now...')
    data_list = []
    exact_match_cnt = 0
    for i, cms_obj in enumerate(cms_show_obj_list):
        print("\r",  "match cms percentage:{0}%".format((round(i + 1) * 100 / len(cms_show_obj_list))),  end = "",  flush = True)
        leven_dis = {}
        for imdb_obj in imdb_obj_list:
            # if not matched any word, we can skip this imdb obj directly. It means the leven_dis will be 0.0
            if not cms_obj.word_list & imdb_obj.word_list: # not matched any word
                continue
            leven_dis[imdb_obj.contentid] = 0
            imdb_obj.cleaned_title = list(set(imdb_obj.cleaned_title))
            for cms_title in cms_obj.cleaned_title:
                if not is_title_valid(cms_title):
                    continue
                for imdb_title in imdb_obj.cleaned_title:
                    if not is_title_valid(imdb_title):
                        continue
                    if imdb_title  == cms_title:
                        cms_obj.exact_match_id.append(imdb_obj.contentid) # match show
                        leven_dis[imdb_obj.contentid] = 1.0
                    else:
                        if len(imdb_title) > 5 and len(cms_title) > 5 and (imdb_title in cms_title or cms_title in imdb_title) and (Levenshtein.ratio(cms_title, imdb_title) > 0.5):
                            cms_obj.fuzzy_match_id.append(imdb_obj.contentid)
                        leven_dis[imdb_obj.contentid] = max(leven_dis[imdb_obj.contentid], Levenshtein.ratio(cms_title, imdb_title))
        # find match id list, we prior choose the exact match ids
        if len(cms_obj.exact_match_id) > 0:
            cms_obj.matched = True
            exact_match_cnt += 1
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
            temp_score = 0.0
            if match_id in id_to_basics_ptitle:
                #match year
                if is_valid(cms_obj.startyear) & is_valid(id_to_basics_year[match_id][0]):
                    if str(cms_obj.startyear) == id_to_basics_year[match_id][0]:
                        temp_score  += 5.0

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
        
        # write to the excel
        if len(cms_obj.final_match_id) > 0:
            for match_id in cms_obj.final_match_id:
                data_list.append(
                        {
                            'exact_matched': cms_obj.matched,
                            'match_actors': cms_obj.match_actors,
                            'match_director': cms_obj.match_director,
                            'cms_title': cms_obj.title,
                            'imdb_title': ','.join(id_to_basics_otitle[match_id]),
                            'cms_directors': ','.join(cms_obj.directors),
                            'imdb_directors': id_to_directors[match_id],
                            'cms_startyear': cms_obj.startyear,
                            'imdb_startyear': id_to_basics_year[match_id][0],
                            'cms_endyear': cms_obj.endyear,
                            'imdb_endyear': id_to_basics_endyear[match_id][0],
                            'cms_actors': ','.join(cms_obj.actors),
                            'imdb_actors': id_to_actors[match_id],
                            'cms_contentid': cms_obj.contentid,
                            'imdb_titleid': match_id
                            }
                        )
        else:
            data_list.append(
                        {
                            'exact_matched': cms_obj.matched,
                            'match_actors': cms_obj.match_actors,
                            'match_director': cms_obj.match_director,
                            'cms_title': cms_obj.title,
                            'imdb_title': '',
                            'cms_directors': ','.join(cms_obj.directors),
                            'imdb_directors': '',
                            'cms_startyear': cms_obj.startyear,
                            'imdb_startyear': '',
                            'cms_endyear': cms_obj.endyear,
                            'imdb_endyear': '',
                            'cms_actors': ','.join(cms_obj.actors),
                            'imdb_actors': '',
                            'cms_contentid': cms_obj.contentid,
                            'imdb_titleid': ''
                            }
                        )
    data = pd.DataFrame(data_list, columns = ['exact_matched', 'match_director', 'match_actors', 'cms_title', 'imdb_title', 'cms_directors', 'imdb_directors', 'cms_startyear', 'imdb_startyear', 'cms_endyear', 'imdb_endyear', 'cms_actors', 'imdb_actors',
        'cms_contentid','imdb_titleid'])
    data.to_excel(save_excel_name, index=False)
    end_1 = timer()
    print(exact_match_cnt)
    print(end_1 - end)
    