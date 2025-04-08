import torch
import torch.multiprocessing as mp


def sender(queue):
    tensor = torch.randn(3, 3, device='cuda')
    print("Sender tensor:", tensor)

    # Send using CUDA IPC
    queue.put(tensor)


def receiver(queue):
    tensor = queue.get()
    print("Receiver tensor:", tensor)


if __name__ == '__main__':
    mp.set_start_method('spawn')  # or 'fork' depending on platform
    queue = mp.Queue()

    p1 = mp.Process(target=sender, args=(queue,))
    p2 = mp.Process(target=receiver, args=(queue,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
