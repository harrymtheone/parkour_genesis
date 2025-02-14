from rich.console import Console
from rich.table import Table

console = Console()

# Create a table
table = Table(show_header=True, show_lines=True)

# First row (single column spanning all three columns)
table.add_column(f'time: {env.episode_length_buf[env.lookat_id].item() / 50:.2f}')
table.add_column("vx")
table.add_column("vy")
table.add_column("yaw")
table.add_column("target_yaw")

table.add_row("cmd", )
table.add_row("real", )
