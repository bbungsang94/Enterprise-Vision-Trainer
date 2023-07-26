import inspect
from typing import Tuple


def example_function() -> (int, int, int):
    return 1, 2, 3

def get_return_variable_count(func):
    signature = inspect.signature(func)
    return len(signature.return_annotation)

# 예시 함수를 호출하여 반환 변수의 개수를 확인합니다.
result = example_function()
return_variable_count = get_return_variable_count(example_function)

print("함수의 반환 변수 개수:", return_variable_count)
print("함수의 반환 값:", result)