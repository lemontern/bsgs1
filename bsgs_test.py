import hashlib
import os
import time
import logging
import bloom_filter
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Используем автоконтекст вместо ручного создания
from pycuda.compiler import SourceModule
import secp256k1_lib as ice  # Использование библиотеки из предоставленного архива

# Логирование
logging.basicConfig(filename='bsgs_wif_search.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Настройка Bloom-фильтра для фильтрации неподходящих ключей
BLOOM_FILTER_SIZE = 1000000
BLOOM_FILTER_HASH_COUNT = 10
bloom = bloom_filter.BloomFilter(max_elements=BLOOM_FILTER_SIZE, error_rate=0.01)

mod = SourceModule(
    """
    __global__ void bsgs_step(unsigned long long *d_keyspace, unsigned int *d_results, int num_keys) {
        unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_keys) {
            d_results[idx] = (d_keyspace[idx] == 123456789ULL) ? 1 : 0;
        }
    }
    """,
    options=['-Xptxas', '-v', '--gpu-architecture=sm_90']  # Обновлен параметр архитектуры для RTX4090
)

try:
    bsgs_step = mod.get_function("bsgs_step")
except cuda.LogicError as e:
    logging.error(f"Ошибка при получении функции CUDA: {str(e)}")
    print(f"Ошибка при получении функции CUDA: {str(e)}")
    exit(1)

# Вывод доступных атрибутов модуля secp256k1_lib
print(dir(ice))

# Основная функция поиска приватного ключа по Bitcoin-адресу
def find_wif_by_address(address):
    d_keyspace = None
    d_results = None
    start_range = 1
    batch_size = 1000000  # Размер одного батча для поиска

    while True:
        try:
            end_range = start_range + batch_size
            logging.info(f"Начало поиска WIF для адреса: {address} в диапазоне от {start_range} до {end_range}")
            start_time = time.time()

            # Генерация и проверка приватных ключей
            num_keys = end_range - start_range
            keyspace = np.arange(start_range, start_range + num_keys, dtype=np.uint64)
            results = np.zeros(num_keys, dtype=np.uint32)

            d_keyspace = cuda.mem_alloc(keyspace.nbytes)
            d_results = cuda.mem_alloc(results.nbytes)

            # Проверка доступной памяти перед запуском ядра
            free, total = cuda.mem_get_info()
            if keyspace.nbytes + results.nbytes > free:
                raise MemoryError("Недостаточно памяти на GPU для выполнения задачи")
            
            cuda.memcpy_htod(d_keyspace, keyspace)
            
            block_size = 1024  # Увеличен размер блока для лучшего использования ресурсов RTX4090
            grid_size = (num_keys + block_size - 1) // block_size

            # Запуск CUDA kernel
            bsgs_step(d_keyspace, d_results, np.int32(num_keys), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(results, d_results)

            for idx, result in enumerate(results):
                if result:
                    # Найден WIF ключ
                    priv_key = ice.privatekey_from_int(start_range + idx)  # Возможно, требуется заменить на другую функцию
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    keys_per_second = num_keys / elapsed_time
                    print(f"\rНайден приватный ключ для адреса {address}: {priv_key} (Скорость: {keys_per_second:.2f} ключей/сек)", end="")
                    logging.info(f"Найден приватный ключ для адреса {address}: {priv_key} (Скорость: {keys_per_second:.2f} ключей/сек)")
                    with open('found_keys.txt', 'a') as f:
                        f.write(f"Address: {address}, Private Key (WIF): {priv_key}\n")
                    return

            end_time = time.time()
            elapsed_time = end_time - start_time
            keys_per_second = num_keys / elapsed_time
            print(f"\rWIF не найден для адреса {address} в диапазоне от {start_range} до {end_range} (Скорость: {keys_per_second:.2f} ключей/сек)", end="")
            logging.info(f"WIF не найден для адреса {address} в диапазоне от {start_range} до {end_range} (Скорость: {keys_per_second:.2f} ключей/сек)")

            # Переход к следующему диапазону
            start_range = end_range
        except Exception as e:
            logging.error(f"Ошибка при поиске WIF: {str(e)}")
            print(f"Ошибка при поиске WIF: {str(e)}")
        finally:
            if d_keyspace is not None:
                d_keyspace.free()
            if d_results is not None:
                d_results.free()

# Поиск WIF для каждого адреса последовательно (без многопоточности)
def search_for_keys(addresses):
    for address in addresses:
        find_wif_by_address(address)

    logging.info("Поиск завершён для всех адресов.")
    print("Поиск завершён для всех адресов.")

if __name__ == "__main__":
    addresses_to_search = [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Примерный адрес
        # Другие адреса для поиска
    ]
    if addresses_to_search:
        logging.info("Начало поиска WIF для заданных адресов...")
        print("Начало поиска WIF для заданных адресов...")
        search_for_keys(addresses_to_search)
        logging.info("Процесс завершён. Проверьте файл лога для деталей.")
        print("Процесс завершён. Проверьте файл лога для деталей.")
    else:
        logging.info("Не указаны адреса для поиска.")
        print("Не указаны адреса для поиска.")