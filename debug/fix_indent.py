import re

file_path = "config/universe.py"
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("            # For company name"):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            company_names = "):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            company_name = "):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            # We can use "):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            aliases = "):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            # Remove Ltd"):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            short_name = "):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("            if short_name"):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("                aliases.append"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("            universe[sym] = {"):
        new_lines.append(line.replace("            ", "        ", 1))
    elif line.startswith("                \"ticker\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("                \"nse_symbol\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("                \"company_name\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("                \"name_aliases\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("                \"sector\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("                \"n_insider_filings\":"):
        new_lines.append(line.replace("                ", "            ", 1))
    elif line.startswith("            }"):
        new_lines.append(line.replace("            ", "        ", 1))
    else:
        new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
