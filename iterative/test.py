class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person({self.name}, {self.age})"

    def __eq__(self, other):
        if isinstance(other, Person):
            return self.name == other.name and self.age == other.age
        return False
    
    def __hash__(self):
        return hash((self.name, self.age))
    
    def __iter__(self):
        yield self.name
        yield self.age
    
    def __getitem__(self, index):
        if index == 0:
            return self.name
        elif index == 1:
            return self.age
        else:
            raise IndexError("Index out of range")
        
    