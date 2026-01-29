from math import sqrt


class VectorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_cords(self):
        return self.x, self.y

    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __add__(self, other):
        if isinstance(other, tuple):
            return self.x + other[0], self.y + other[1]
        if isinstance(other, Vector):
            return self.x + other.x, self.y + other.y
        raise VectorError('Ошибка в сложении векторов')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other)
        raise VectorError('Ошибка в умножении вектора на число')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return self.__add__(other.__mul__(-1))
        if isinstance(other, tuple):
            return self.__add__(Vector(*other).__mul__(-1))
        raise VectorError('Ошибка в вычитании векторов')

    def __rsub__(self, other):
        if isinstance(other, set):
            new_set = {element for element in other if element != self}
            return new_set
        if isinstance(other, list):
            new_list = [element for element in other if element != self]
            return new_list
        if isinstance(other, tuple):
            return Vector(*other).__add__(self.__mul__(-1))
        raise VectorError('Ошибка в вычитании векторов')

    def __neg__(self):
        return self.__mul__(-1)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.x / other, self.y / other
        raise VectorError('Ошибка в делении вектора на число')

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return self.x // other, self.y // other
        raise VectorError('Ошибка в делении вектора на число')

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1]
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, tuple):
            return self.length() < Vector(*other).length()
        if isinstance(other, Vector):
            return self.length() < other.length()
        raise VectorError('Ошибка в сравнении вектора')

    def __gt__(self, other):
        if isinstance(other, tuple):
            return self.length() > Vector(*other).length()
        if isinstance(other, Vector):
            return self.length() > other.length()
        raise VectorError('Ошибка в сравнении вектора')

    def __le__(self, other):
        if isinstance(other, tuple):
            return self.length() <= Vector(*other).length()
        if isinstance(other, Vector):
            return self.length() <= other.length()
        raise VectorError('Ошибка в сравнении вектора')

    def __ge__(self, other):
        if isinstance(other, tuple):
            return self.length() >= Vector(*other).length()
        if isinstance(other, Vector):
            return self.length() >= other.length()
        raise VectorError('Ошибка в сравнении вектора')

    def __hash__(self):
        return hash((self.x, self.y))

    def __and__(self, other):
        if isinstance(other, set):
            new_set = {tuple(element) for element in other if element == self}
            return new_set
        raise VectorError('Пересечения применимы только к множествам \'set\'')

    def __rand__(self, other):
        return self.__and__(other)


if __name__ == "__main__":
    V1 = Vector(1, 2)
    V2 = Vector(2, -2)
    D1 = ((0, 0), (1, 2)) & V1
    D2 = V1 >= V1
    D3 = V1 - V2
    print(D1, D2, D3)





































