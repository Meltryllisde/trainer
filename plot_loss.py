"""Plot loss curves from two training runs (with/without IoU reward)."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_loss(path):
    steps, losses = [], []
    if not os.path.exists(path):
        return steps, losses
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
    return steps, losses


def moving_average(values, window):
    """Centered moving average; output length == len(values)."""
    if not values:
        return []
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def ema(values, alpha=0.1):
    """Exponential moving average."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def main():
    s1, l1 = load_loss('loss_with_iou.csv')
    s2, l2 = load_loss('loss_no_iou.csv')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw + moving average (window=20)
    ax = axes[0]
    if l1:
        ax.plot(s1, l1, color='tab:blue', alpha=0.25, linewidth=1)
        ax.plot(s1, moving_average(l1, 20), color='tab:blue',
                linewidth=2, label='with IoU (MA-20)')
    if l2:
        ax.plot(s2, l2, color='tab:orange', alpha=0.25, linewidth=1)
        ax.plot(s2, moving_average(l2, 20), color='tab:orange',
                linewidth=2, label='no IoU (MA-20)')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_title('Moving Average (window=20)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: EMA(alpha=0.05) for stronger smoothing
    ax = axes[1]
    if l1:
        ax.plot(s1, ema(l1, 0.05), color='tab:blue',
                linewidth=2, label='with IoU (EMA α=0.05)')
    if l2:
        ax.plot(s2, ema(l2, 0.05), color='tab:orange',
                linewidth=2, label='no IoU (EMA α=0.05)')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_title('EMA Smoothed (α=0.05)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle('Training Loss: IoU reward vs. no IoU reward', fontsize=14)
    plt.tight_layout()
    out = 'loss_curves.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')

    def summary(name, l):
        if not l:
            print(f'  {name}: empty')
            return
        n = len(l)
        first = sum(l[:20]) / min(20, n)
        last = sum(l[-20:]) / min(20, n)
        print(f'  {name}: steps={n} first20_mean={first:.4f} last20_mean={last:.4f} '
              f'delta={last-first:+.4f}')

    summary('with_iou', l1)
    summary('no_iou  ', l2)


if __name__ == '__main__':
    main()
