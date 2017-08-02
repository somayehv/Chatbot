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
        self.brand_to_categories_map = {}
        self.product_to_brand_map = {}
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
        self.found_brands = set()
        self.found_product_key_words = set()
        self.found_product_names = set()
        self.possible_product_names = set()

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
                    if brand not in self.brand_to_categories_map:
                        self.brand_to_categories_map[brand] = set()
                    self.brand_to_categories_map[brand].add(category)
                    self.product_to_brand_map[product] = brand
        self.make_key_word_to_category_map()
        self.make_key_word_to_product_map()

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
            key_words = [word for word, word_pos in pos if (word_pos in ['NN', 'NNP', 'NNS', 'NNPS'] and len(word) > 2)]
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
            self.found_brands = {word for word in words if word in self.brands}
        if any(word in self.product_key_words for word in words):
            self.found_product_key_words = {word for word in words if word in self.product_key_words}
        if any(word in self.product_names for word in words):
            self.found_product_names = {word for word in words if word in self.product_names}

    def generate_response(self):
        if any(self.sentence == product_name for product_name in self.product_names):
            self.found_product_names = {self.sentence}
            return self.offer_prices()
        if self.sentence == 'exit':
            return self.GOODBYE
        if self.sentence == 'reset':
            self.reset()
            return self.WELCOME_BACK
        self.found_categories.update([self.key_word_to_category_map[key_word]
                                      for key_word in self.found_category_key_words])
        if self.found_product_names:
            return self.offer_prices()
        if self.possible_product_names:
            return self.offer_prices_based_on_possible_product_names()
        response = self.DEFAULT_RESPONSE
        if self.found_product_key_words:
            response = self.suggest_product_names_from_key_words()
        elif self.found_brands and not self.found_categories:
            response = self.suggest_categories()
        elif self.found_categories and not self.found_brands:
            response = self.suggest_brands()
        elif self.found_category_key_words and self.found_brands:
            response = self.suggest_product_names_from_categories_and_brands()
        return response

    def reset(self):
        self.found_categories = set()
        self.found_category_key_words = set()
        self.found_brands = set()
        self.found_product_key_words = set()
        self.found_product_names = set()
        self.possible_product_names = set()

    def offer_prices_based_on_possible_product_names(self):
        tokens = nltk.word_tokenize(self.sentence)
        names_with_intersections = set()
        for name in self.possible_product_names:
            temp_tokens = nltk.word_tokenize(name)
            if set(temp_tokens).intersection(tokens):
                names_with_intersections.add(name)
        if names_with_intersections:
            self.found_product_names = names_with_intersections
        else:
            self.found_product_names = self.possible_product_names
        return self.offer_prices()

    def offer_prices(self):
        response = ''
        for product_name in self.found_product_names:
            response += '{} will cost {} of rent per month.'.format(
                product_name, self.product_name_to_price_map[product_name]) + '\n'
        return response

    def suggest_categories(self):
        response = ''
        if len(self.found_brands) > 1:
            response = 'Which one of the brands you mentioned are you interested in?'
            for brand in self.found_brands:
                response += '\n' + brand
        elif len(self.found_brands) == 1:
            brand = list(self.found_brands)[0]
            categories = self.brand_to_categories_map[brand]
            if len(categories) == 1:
                self.found_categories.add(list(categories)[0])
                response = self.suggest_product_names_from_categories_and_brands()
            else:
                response = 'I understand you are interested in the brand {}. ' \
                           'Which of the following categories are you interested in?'.format(brand)
                for category in categories:
                    response += '\n' + category
        return response

    def suggest_product_names_from_key_words(self):
        possible_product_names = set()
        for key_word in self.found_product_key_words:
            possible_product_names.update(self.key_word_to_product_map[key_word])
        possible_brands = set()
        possible_brands.update([self.product_to_brand_map[product_name] for product_name in possible_product_names])
        brands = possible_brands.intersection(self.found_brands)
        if len(possible_product_names) == 1:
            self.found_product_names.update(possible_product_names)
            response = 'We offer the following product:'
            response = response + '\n' + self.offer_prices()
        elif len(possible_brands) == 1:
            response = self.suggest_product_names_from_list(possible_product_names)
        elif self.found_brands and len(brands) == 1:
            possible_product_names = [name for name in possible_product_names
                                      if self.product_to_brand_map[name] in brands]
            if len(possible_product_names) == 1:
                self.found_product_names.update(possible_product_names)
                response = 'We offer the following product:'
                response = response + '\n' + self.offer_prices()
            else:
                response = self.suggest_product_names_from_list(possible_product_names)
        else:
            response = 'Which of the following brands are you interested in?'
            for brand in possible_brands:
                response += '\n' + brand
        return response

    def suggest_product_names_from_list(self, products):
        response = 'Which of the following products are you interested in?'
        for product in products:
            response += '\n' + product
            self.possible_product_names.add(product)
        return response

    def suggest_product_names_from_categories_and_brands(self):
        if len(self.found_brands) == 1:
            brand = list(self.found_brands)[0]
            if len(self.brand_to_categories_map[brand]) == 1:
                response = \
                    self.suggest_product_names_from_one_category_and_one_brand(brand, self.brand_to_products_map[brand])
            else:
                response = self.suggest_product_names_from_multiple_categories_and_one_brand(brand)
        else:
            response = 'Which of the following brands are you interested in?'
            for brand in self.found_brands:
                response = response + '\n' + brand
        return response

    def suggest_product_names_from_one_category_and_one_brand(self, brand, products):
        if len(self.brand_to_products_map[brand]) == 1:
            product_name = self.brand_to_products_map[brand][0]
            self.found_product_names.add(product_name)
            response = 'We offer the following product:'
            response += '\n' + self.offer_prices()
        else:
            response = self.suggest_product_names_from_list(products)
        return response

    def suggest_product_names_from_multiple_categories_and_one_brand(self, brand):
        possible_categories = [cat for cat in self.brand_to_categories_map[brand]]
        intersection_of_categories = self.found_categories.intersection(possible_categories)
        if len(intersection_of_categories) == 1:
            category = list(intersection_of_categories)[0]
            response = \
                self.suggest_product_names_from_one_category_and_one_brand(brand, self.store_data[category][brand])
        elif len(intersection_of_categories) == 0:
            response = 'Which of the following categories are you interested in?'
            for category in possible_categories:
                response += '\n' + category
        else:
            response = 'Which of the following categories are you interested in?'
            for category in intersection_of_categories:
                response += '\n' + category
        return response

    def suggest_brands(self):
        if len(self.found_categories) == 1:
            category = list(self.found_categories)[0]
            response = 'I understand that you are interested in the category {}\n'.format(category)
            brands = list(self.store_data[category].keys())
            if len(brands) == 1:
                brand = brands[0]
                products = self.brand_to_products_map[brand]
                if len(products) == 1:
                    product = products[0]
                    self.found_product_names.add(product)
                    response += self.offer_prices()
                else:
                    response = self.suggest_product_names_from_list(products)
            else:
                response += 'We offer the following brands:'
                for brand in self.store_data[category]:
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
    print(chatbot.GREETING)
    while chatbot.sentence != 'exit':
        chatbot.sentence = input().lower()
        chatbot.update_key_words()
        print(chatbot.generate_response())
