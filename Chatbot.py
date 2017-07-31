import nltk
import csv
from nltk.stem import *


class ChatBot:

    GREETING = 'Hi there! Welcome to Grover!\nWhich category, brand, or product are you looking for?\n' \
               '(You can type \'reset\' at any time to start over and \'exit\' if you are done.)'
    WELCOME_BACK = 'Welcome back!\nWhat other category, brand, or product are you looking for?'
    DEFAULT_RESPONSE = 'Sorry but I am not sure which category, brand, or product you are looking for.'
    GOODBYE = 'Bye! Hope you visit us again soon!'

    def __init__(self):
        self.store_data = {}
        self.brand_to_products_map = {}
        self.product_names = []
        self.brands = set()
        self.categories = set()
        self.product_name_to_price_map = {}
        self.product_key_words = set()
        self.category_key_words = set()
        self.key_word_to_category_map = {}
        self.key_word_to_product_map = {}
        self.category, self.brand, self.product_name = None, None, None
        self.sentence = ''
        self.found_category_key_words = set()
        self.found_categories = set()
        self.found_brand_key_words = set()
        self.found_product_key_words = set()
        self.found_product_names = set()

    def extract_data_from_file(self, file_name):
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] != 'Product Id':
                    product = row[1].lower()
                    self.product_names.append(product)
                    brand = row[2].lower()
                    self.brands.add(brand)
                    category = row[3].lower()
                    self.categories.add(category)
                    price = row[4].lower()
                    if category not in self.store_data:
                        self.store_data[category] = {}
                    if brand not in self.store_data[category]:
                        self.store_data[category][brand] = {}
                    if brand not in self.brand_to_products_map:
                        self.brand_to_products_map[brand] = []
                    self.brand_to_products_map[brand].append(product)
                    self.store_data[category][brand][product] = price
                    self.product_name_to_price_map[product] = price

    def make_key_word_to_category_map(self):
        for category in self.categories:
            temp = category.split('&')
            for word in temp:
                self.key_word_to_category_map[word.strip()] = category
                stem = stemmer.stem(word.strip())
                if stem not in self.key_word_to_category_map:
                    self.key_word_to_category_map[stem] = category
        self.category_key_words.update(self.key_word_to_category_map.keys())

    def make_key_word_to_product_map(self):
        for product in self.product_names:
            tokens = nltk.word_tokenize(product)
            pos = nltk.pos_tag(tokens)
            key_words = [word for word, word_pos in pos if word_pos in ['NN', 'NNP', 'NNS', 'NNPS']]
            for key_word in key_words:
                if key_word not in self.key_word_to_product_map:
                    self.key_word_to_product_map[key_word] = set()
                self.key_word_to_product_map[key_word].add(product)
        self.product_key_words.update(self.key_word_to_product_map.keys())

    def update_key_words(self):
        tokens = nltk.word_tokenize(self.sentence)
        two_words = [tokens[i] + ' ' + tokens[i + 1] for i in range(len(tokens) - 1)]
        words = tokens
        words.extend(two_words)
        if any(word in self.category_key_words for word in words):
            self.found_category_key_words = {word for word in words if word in self.category_key_words}
        if any(word in self.categories for word in words):
            self.found_categories = {word for word in words if word in self.categories}
        if any(word in self.brands for word in words):
            self.found_brand_key_words = {word for word in words if word in self.brands}
        if any(word in self.product_key_words for word in words):
            self.found_product_key_words = {word for word in words if word in self.product_key_words}
        if any(word in self.product_names for word in words):
            self.found_product_names = {word for word in words if word in self.product_names}

    def generate_response(self):
        response = self.DEFAULT_RESPONSE
        if any(self.sentence == product_name for product_name in self.product_names):
            self.found_product_names = {self.sentence}
            response = self.offer_prices()
        if self.sentence == 'exit':
            response = self.GOODBYE
        elif self.sentence == 'reset':
            self.found_categories = set()
            self.found_category_key_words = set()
            self.found_brand_key_words = set()
            self.found_product_key_words = set()
            self.found_product_names = set()
            response = self.WELCOME_BACK
        elif self.found_product_names:
            response = self.offer_prices()
        elif self.found_product_key_words:
            response = self.suggest_product_names_from_key_words()
        elif self.found_brand_key_words:
            response = self.suggest_product_names_from_brands()
        elif self.found_categories:
            response = self.suggest_brands()
        elif self.found_category_key_words:
            self.found_categories.update([self.key_word_to_category_map[key_word]
                                          for key_word in self.found_category_key_words])
            response = self.suggest_brands()
        return response

    def offer_prices(self):
        response = ''
        for product_name in self.found_product_names:
            response += '{} will cost {} of rent per month.'.format(
                product_name, self.product_name_to_price_map[product_name]) + '\n'
        return response

    def suggest_product_names_from_key_words(self):
        for key_word in self.found_product_key_words:
            self.found_product_names.update(self.key_word_to_product_map[key_word])
        if len(self.found_product_names) == 1:
            response = 'We offer the following product:'
            response = response + '\n' + self.offer_prices()
        else:
            response = 'Which of the following products do you want? (Please write only the exact product name.)'
            for product in self.found_product_names:
                response += '\n' + product
        return response

    def suggest_product_names_from_brands(self):
        if len(self.found_brand_key_words) == 1:
            if len(self.brand_to_products_map[list(self.found_brand_key_words)[0]]) == 1:
                self.found_product_names.add(self.brand_to_products_map[list(self.found_brand_key_words)[0]][0])
                response = 'We offer the following product:'
                response = response + '\n' + self.offer_prices()
            else:
                response = 'Which of the following products do you want? (Please write only the exact product name.)'
                for product in self.brand_to_products_map[list(self.found_brand_key_words)[0]]:
                    response = response + '\n' + product
        else:
            response = 'We offer the following brands:'
            for brand in self.found_brand_key_words:
                response = response + '\n' + brand
        return response

    def suggest_brands(self):
        if len(self.found_categories) == 1:
            response = 'I understand that you are interested in the category {}'.format(list(self.found_categories)[0])
            if len(self.store_data[list(self.found_categories)[0]]) == 1:
                if len(self.brand_to_products_map[self.store_data[list(self.found_categories)[0]][0]]) == 1:
                    self.found_product_names.add(
                        self.brand_to_products_map[self.store_data[list(self.found_categories)[0]][0]][0])
                    response += '\n' + self.offer_prices()
                else:
                    response += \
                        '\nWhich of the following products do you want? (Please write only the exact product name.)'
                    for product in self.brand_to_products_map[self.store_data[list(self.found_categories)[0]][0]]:
                        response += '\n' + product
            else:
                response += '\nWe offer the following brands:'
                for brand in self.store_data[list(self.found_categories)[0]]:
                    response += '\n' + brand
        else:
            response = ''
            for category in self.found_categories:
                response += '\nFor category {} we offer the following brand(s):'.format(category)
                for brand in self.store_data[category]:
                    response += '\n' + brand
        return response


if __name__ == "__main__":
    stemmer = PorterStemmer()
    chatbot = ChatBot()
    chatbot.extract_data_from_file('Data.csv')
    chatbot.make_key_word_to_category_map()
    chatbot.make_key_word_to_product_map()
    print(chatbot.GREETING)
    while chatbot.sentence != 'exit':
        chatbot.sentence = input().lower()
        chatbot.update_key_words()
        print(chatbot.generate_response())
