from abc import ABC, abstractmethod

class Goods(ABC):
    def __init__(self, price, name):
        self.price = price
        self.name = name
    
    def get_price(self):
        return self.price
    
    def get_name(self):
        return self.name

    @abstractmethod
    def get_marketing_info(self):
        pass

class InfoManager():
    def __init__(self, item: Goods):
        self.item = item
    
    def marketing_info(self):
        return self.item.get_marketing_info()

class Liquid(Goods):
    def __init__(self, price, name, volume):
        super().__init__(price, name)
        self.volume = volume

    def get_marketing_info(self):
        return f'{self.name}, {self.volume} ml: ${self.price}'
        
        
class Snakes(Goods):
    def __init__(self, price, name, mass):
        super().__init__(price, name)
        self.mass = mass


    def get_marketing_info(self):
        return f'{self.name}, {self.mass} g: ${self.price}'        



class PlaceHolder():
    CAPACITY = 10
    
    def __init__(self, index):
        self.id = index
        self.qty = 0
        self.items = []
        
    def load(self, item):
    
        if self.qty == self.CAPACITY:
            return None   # placeholder full
        
        if self.qty !=0:   # placeholder has to hold only similar type of items
            if type(self.items[-1]) != type(item):
                return None
        
        self.items.append(item)
        self.qty += 1
        
    def take(self):
        if self.qty == 0:
            return None   # placeholder is empty

        self.qty -= 1
        return self.items.pop()
    
    def get_qty(self):
        return self.qty
    
    def get_state(self):
        return self.items
    


class ShelfModule():
    SHELF_SIZE = 9
    
    def __init__(self):
        self.shelf = []
        
        for i in range(self.SHELF_SIZE):
            self.shelf.append(PlaceHolder(i))
            
    def load(self, index, goods):
        [self.shelf[index].load(g) for g in goods]
        
    def take(self, index):
        return self.shelf[index].take()
    
    def get_state(self):
        return self.shelf
    

class CashModule():
    
    def __init__(self, initial_cash):
        self.cash_inside = initial_cash
        self.cash_account = 0
        
    def put_money(self, amount):
        self.cash_account += amount
        
    def get_resedual(self):
        resedual = self.cash_account
        self.cash_account = 0
        return resedual
    
    def pay(self, val):
        if val > self.cash_account:
            return False
        
        self.cash_account -= val
        self.cash_inside += val
        
    def get_account(self):
        return self.cash_account
    
    def get_state(self):
        return (self.cash_inside, self.cash_account)
    
    
class VendingMachine():
    
    def __init__(self, initial_cash):
        self.shelf_module = ShelfModule()
        self.cash_module = CashModule(initial_cash)
    
    def add_items(self, place_ids, items):
        
        for i, its in zip(place_ids, items):
            
            self.shelf_module.load(i, its)
    
    def add_money(self, amount):
        self.cash_module.put_money(amount)
        return amount
    
    def get_resedual(self):
        return self.cash_module.get_resedual()
    
    def buy_item(self, index):
        if self.shelf_module.get_state()[index].get_qty() == 0:
            return None
        
        price_item = self.shelf_module.get_state()[index].get_state()[-1].get_price()
        
        if self.cash_module.get_account() > price_item:
            self.cash_module.pay(price_item)
            return self.shelf_module.take(index)
    
    def showcase(self):
        res = ''
        res += 'Total INFO: \n  Shelves:\n'
        for i, ph in enumerate(self.shelf_module.get_state()):
            itms = ph.get_state()
            if len(itms) == 0:
                res += f'    {i}: Empty\n'
            else:
                res += f'    {i}: {itms[-1].get_marketing_info()} x{len(itms)}\n'
        cash_inside, cash_account = self.cash_module.get_state()
        res += f'  Cash  inside : ${cash_inside}\n'
        res += f'  Cash account : ${cash_account}\n'        
        return res
    
if __name__ == '__main__':
    pepsi    = Liquid(10, 'Pepsi',          '0.5')
    fanta    = Liquid(12, 'Fanta',          '0.5')
    estrella = Snakes(22, 'Estrella chips', '300')
    lays     = Snakes(18, 'Lays chips',     '300')

    v = VendingMachine(200)

    print(v.showcase())

    print('Adding goods\n')
    v.add_items([1,2,4,5],[[pepsi, pepsi], [estrella, estrella, estrella],[lays,lays],[fanta, fanta]])

    print(v.showcase())

    print('Adding money: $', v.add_money(15),'\n')

    print(v.showcase())

    print('Buying: ', v.buy_item(1).get_marketing_info(),'\n')

    print(v.showcase())

    print('Ger residual: $', v.get_resedual(),'\n')

    print(v.showcase())