import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns


def load_and_prepare_data(metadata_file, image_folder, json_file):
    # Load metadata
    metadata = pd.read_csv(metadata_file, encoding='utf-8')

    # Normalize ID, standardize 'Age of bruise', and normalize 'Gender'
    metadata['Normalized_ID'] = metadata['ID'].str.strip().str.lower()
    metadata['Gender'] = metadata['Gender'].str.lower().replace({'w': 'female', 'm': 'male'})
    metadata['Age of bruise'] = metadata['Age of bruise'].str.strip().str.lower().replace({'frisch ': 'frisch', 'älter ': 'älter'})

    # Function to convert age descriptions into a numeric format (years)
    def convert_age(value):
        try:
            return int(value.split()[0]) / 12.0 if 'Monat' in value else int(value)
        except ValueError:
            return None

    # Apply function, clean data, and filter
    metadata['Age'] = metadata['Age'].apply(convert_age)
    metadata = metadata.dropna(subset=['Age'])

    # Load JSON file with specific case IDs to keep
    with open(json_file, 'r') as file:
        keep_ids_dict = json.load(file)

    # Modify the metadata dataframe to include a flag for IDs to keep
    # Normalize case/folder names as they appear in the JSON
    metadata['Case/Folder'] = metadata['Case/Folder'].str.strip().str.lower()
    
    # Check if each row's Case/Folder is in the JSON dictionary
    metadata['Keep'] = metadata.apply(
        lambda row: row['Normalized_ID'] in keep_ids_dict.get(row['Case/Folder'], []) or row['Case/Folder'] not in keep_ids_dict,
        axis=1
    )

    # Filter metadata based on this flag
    filtered_metadata = metadata[metadata['Keep']]

    return filtered_metadata



def plot_age_distribution_by_sex(filtered_metadata, output_folder, minrange, maxrange):
    # Get actual min and max ages in the data
    actual_min_age = filtered_metadata['Age'].min()
    actual_max_age = filtered_metadata['Age'].max()
    
    # Adjust minrange and maxrange if out of bounds
    if minrange < actual_min_age:
        minrange = actual_min_age
    if maxrange > actual_max_age:
        maxrange = actual_max_age

    # Filter data based on adjusted age range
    age_filtered_metadata = filtered_metadata[(filtered_metadata['Age'] >= minrange) & (filtered_metadata['Age'] <= maxrange)]
    
    # Group data by case and gender
    grouped_metadata = age_filtered_metadata.groupby(['Case/Folder', 'Gender']).agg({'Age': 'first'})

    # Define age bins
    bin_step = (maxrange - minrange) / 5
    age_bins = [minrange + bin_step * i for i in range(6)]

    # Colorblind-friendly palette
    colors = {'female': '#377eb8', 'male': '#e41a1c'}

    # Plot histograms
    plt.figure(figsize=(12, 8))
    for gender, data in grouped_metadata.groupby('Gender'):
        plt.hist(data['Age'], bins=age_bins, color=colors.get(gender, '#999999'), alpha=0.7, label=f'{gender.capitalize()} (n={data.shape[0]})')

    plt.title(f'Age Distribution by Sex: Ages {int(minrange)} to {int(maxrange)}')
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.legend(title='Sex')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, f"age_distribution_by_sex_{int(minrange)}_to_{int(maxrange)}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_hematoma_age_distribution(df, min_age, max_age, output_folder):
    # Filter data based on age range
    age_filtered_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
    
    # Summarize data by hematoma age category
    sizes = age_filtered_df['Age of bruise'].value_counts()
    
    # Translate German terms to English
    translation = {'frisch': 'fresh', 'älter': 'old'}
    sizes.index = [translation.get(age, age) for age in sizes.index]
    
    # Define adjusted colors
    colors = {'old': '#FFD700', 'fresh': '#4169E1'}  # Gold for old, royal blue for fresh
    plot_colors = [colors[age] for age in sizes.index]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(sizes, autopct="%.1f%%", startangle=90, colors=plot_colors)
    title_text = f'Age of Hematoma: Ages {min_age}-{max_age}'
    plt.title(title_text, size=16)
    plt.legend(title='Age of Hematoma', labels=sizes.index, loc="best")

    # Create a filename that includes the age range
    filename = f"Age_of_hematoma_{min_age}_to_{max_age}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_case_photo_info(filtered_metadata, output_folder):
    # Calculate the count of images per case
    image_counts = filtered_metadata['Case/Folder'].value_counts()

    # Filter cases with more than 4 photos
    cases_more_than_4_photos = image_counts[image_counts > 4]

    # Save the names of the cases to a text file
    output_txt_path = os.path.join(output_folder, "cases_more_than_4_photos.txt")
    cases_more_than_4_photos.index.to_series().to_csv(output_txt_path, index=False, header=False)

    # Create a dictionary of cases with their respective photo names
    photo_dict = {}
    for case_id in cases_more_than_4_photos.index:
        case_photos = filtered_metadata[filtered_metadata['Case/Folder'] == case_id]['Normalized_ID'].tolist()
        photo_dict[case_id] = case_photos

    # Save the dictionary to a JSON file
    output_json_path = os.path.join(output_folder, "case_photos.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(photo_dict, json_file, indent=4)


def plot_hematoma_location_by_gender(df, output_folder, age_min, age_max):
    """
    Plot and save the relative frequencies of hematoma locations by gender within a specific age range.

    Parameters:
    - df: pandas DataFrame with 'Location', 'Gender', and 'Age' columns.
    - output_folder: str, path to the directory where the plot will be saved.
    - age_min: int or float, minimum age for filtering the data.
    - age_max: int or float, maximum age for filtering the data.
    """

    # Filter the dataframe based on the age range
    df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)].copy()
    df.Location=df.Location.str.lower()
    correction_dict = {
        'unterarn': 'unterarm',
        'brust ': 'brust',
        'schultern ': 'schultern',
        'oberschenkel ': 'oberschenkel'
    }

    # Apply the corrections
    df['Location'] = df['Location'].replace(correction_dict)

    # Remove any leading/trailing whitespaces
    df['Location'] = df['Location'].str.strip()

    # Create a dictionary for English equivalents
    english_dict = {
        'oberarm': 'upper arm',
        'unterarm': 'forearm',
        'rücken': 'back',
        'gesäß': 'buttocks',
        'halsvorderseite': 'front neck',
        'oberschenkel': 'thigh',
        'unterschenkel': 'lower leg',
        'hände': 'hands',
        'gesicht': 'face',
        'bauch': 'abdomen',
        'schultern': 'shoulders',
        'genitalien': 'genitals',
        'brust': 'chest',
        'behaarte kopfhaut': 'hairy scalp',
        'füße': 'feet',
        'ohren': 'ears',
        'nacken': 'neck',
        'nr': 'not recorded',
        'knie': 'knee',
        'hals': 'neck'
    }

    # Map the 'Location' column to its English equivalents
    df['Location'] = df['Location'].map(english_dict)

    # Capitalize the first letter of each gender designation for consistency in display
    df['Gender'] = df['Gender'].str.capitalize()


    # Calculate total counts for each gender in the filtered data
    total_women = (df['Gender'] == 'Female').sum()
    total_men = (df['Gender'] == 'Male').sum()

    # Calculate the location counts for each gender
    location_counts = df.groupby(['Location', 'Gender']).size().unstack(fill_value=0)

    # Calculate the relative frequencies
    location_counts['Female'] = location_counts['Female'] / total_women
    location_counts['Male'] = location_counts['Male'] / total_men

    # Sort locations by frequency in women
    #sorted_locations = location_counts['Female'].sort_values(ascending=False).index
    #location_counts_sorted = location_counts.loc[sorted_locations]

    # Reset index for seaborn
    location_counts = df.groupby(['Location', 'Gender']).size().unstack(fill_value=0)
    location_counts['Female'] = location_counts['Female'] / total_women
    location_counts['Male'] = location_counts['Male'] / total_men

    # Ensure both 'Female' and 'Male' have significant representation
    location_counts['Difference'] = abs(location_counts['Female'] - location_counts['Male'])
    significant_locations = location_counts[(location_counts['Female'] > 0) & (location_counts['Male'] > 0)]

    # Select top 8 locations with the largest differences and then sort by female frequencies
    top_locations = significant_locations['Difference'].nlargest(8).index
    location_counts_sorted = significant_locations.loc[top_locations].sort_values(by='Female', ascending=False).reset_index()

    # Melt the dataframe for plotting
    location_counts_melted = location_counts_sorted.melt(id_vars='Location', value_vars=['Female', 'Male'], var_name='Gender', value_name='Relative Frequency')

    # Create the plot
    plt.figure(figsize=(6, 10))
    sns.barplot(data=location_counts_melted, x='Relative Frequency', y='Location', hue='Gender', palette='pastel')

    # Set labels and title
    plt.xlabel('Relative Frequency')
    plt.ylabel('Location of Hematoma')
    plt.title(f'Relative Location of Bruises by Sex (Ages {age_min} to {age_max})')
    plt.legend(title='Sex')

    # Save and show the plot
    output_path = os.path.join(output_folder, f'Relative_Location_of_Hematoma_by_Gender_Ages_{age_min}_to_{age_max}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_photos_per_case_histogram(df, output_folder):
    """
    Plot and save a histogram showing the distribution of photos per case.

    Parameters:
    - df: pandas DataFrame with the case information, assuming there's a 'Case/Folder' column.
    - output_folder: str, path to the directory where the plot will be saved.
    """
    # Group by 'Case/Folder' and count the number of photos per case
    image_counts = df.groupby('Case/Folder').size()

    # Calculate statistics
    total_cases = len(image_counts)
    avg_photos = image_counts.mean()
    std_photos = image_counts.std()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(image_counts, color='skyblue', alpha=0.7)  # Automatic bin size
    plt.title('Distribution of Photos per Case')
    plt.xlabel('Number of Photos')
    plt.ylabel('Number of Cases')

    # Annotations with formatted statistics
    stats_text = (f'Total cases: {total_cases}\n'
                  f'Average photos per case: {avg_photos:.2f} (±{std_photos:.2f})')
    plt.annotate(stats_text, xy=(0.62, 0.9), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='none'))

    # Save the plot to the output folder with 300 dpi
    output_path = os.path.join(output_folder, "photos_per_case_histogram.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_top5_locations_stacked_by_age_with_gender_patterns(filtered_metadata, output_folder, age_ranges):
    """
    Stacked bar plot with top 5 locations.
    - X-axis = location
    - One bar per age group
    - Female/Male stacked within
    - Hatch pattern shows sex
    - Color shows age group
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = filtered_metadata.copy()

    # Preprocessing
    df['Location'] = df['Location'].str.lower().str.strip()
    correction_dict = {
        'unterarn': 'unterarm',
        'brust ': 'brust',
        'schultern ': 'schultern',
        'oberschenkel ': 'oberschenkel'
    }
    df['Location'] = df['Location'].replace(correction_dict)

    english_dict = {
        'oberarm': 'upper arm',
        'unterarm': 'forearm',
        'rücken': 'back',
        'gesäß': 'buttocks',
        'halsvorderseite': 'front neck',
        'oberschenkel': 'thigh',
        'unterschenkel': 'lower leg',
        'hände': 'hands',
        'gesicht': 'face',
        'bauch': 'abdomen',
        'schultern': 'shoulders',
        'genitalien': 'genitals',
        'brust': 'chest',
        'behaarte kopfhaut': 'hairy scalp',
        'füße': 'feet',
        'ohren': 'ears',
        'nacken': 'neck',
        'nr': 'not recorded',
        'knie': 'knee',
        'hals': 'neck'
    }
    df['Location'] = df['Location'].map(english_dict)
    df['Gender'] = df['Gender'].str.capitalize()

    all_data = []
    age_labels = []

    for (age_min, age_max) in age_ranges:
        age_label = f"{age_min}-{age_max}"
        age_labels.append(age_label)

        sub = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)].copy()
        if sub.empty:
            continue

        total = len(sub)

        loc_counts = sub.groupby(['Location', 'Gender']).size().unstack(fill_value=0)
        if 'Female' not in loc_counts.columns:
            loc_counts['Female'] = 0
        if 'Male' not in loc_counts.columns:
            loc_counts['Male'] = 0

        loc_counts['Female_rel'] = loc_counts['Female'] / total
        loc_counts['Male_rel'] = loc_counts['Male'] / total
        loc_counts['Age Group'] = age_label

        all_data.append(loc_counts.reset_index())

    # Combine
    combined = pd.concat(all_data)

    # Find top 5 locations
    loc_totals = combined.groupby('Location')[['Female_rel', 'Male_rel']].sum()
    loc_totals['Total'] = loc_totals['Female_rel'] + loc_totals['Male_rel']
    top5 = loc_totals.sort_values('Total', ascending=False).head(8).index.tolist()

    # Filter
    data = combined[combined['Location'].isin(top5)]

    # Order locations for consistent x-axis
    location_order = top5

    # Set up plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for age groups
    from seaborn import color_palette
    colors = [
        "#4878CF",  # muted blue
        "#D65F5F",  # soft red
        "#6ACC65",  # soft green
        "#CC61B0",  # muted purple
        "#FFA500",  # academic orange
        "#8C8C8C",  # medium gray
        "#56B4E9"   # light blue
    ][:len(age_ranges)]


    # Bar width
    total_groups = len(age_ranges)
    group_width = 0.8
    bar_width = group_width / total_groups

    # Draw bars
    for i, age_label in enumerate(age_labels):
        group_data = data[data['Age Group'] == age_label]

        for j, loc in enumerate(location_order):
            row = group_data[group_data['Location'] == loc]
            if row.empty:
                continue
            female_val = row['Female_rel'].values[0]
            male_val = row['Male_rel'].values[0]
            x = j - group_width/2 + i * bar_width

            # Female = solid
            ax.bar(x, female_val, width=bar_width, color=colors[i], label=age_label if i == 0 else None)
            # Male = striped hatch
            ax.bar(x, male_val, bottom=female_val, width=bar_width, color=colors[i], hatch='///', edgecolor='black')

    ax.set_xticks(range(len(location_order)))
    ax.set_xticklabels(location_order, rotation=45, ha='right')
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel("Location")
    ax.set_title("Top 5 Hematoma Locations by Age Group (Sex Stacked)")

    # Create custom legend
    age_patches = [Patch(facecolor=colors[i], label=age_labels[i]) for i in range(len(age_labels))]
    sex_patches = [Patch(facecolor='gray', label='Female'),
                   Patch(facecolor='gray', hatch='///', edgecolor='black', label='Male')]
    legend1 = ax.legend(handles=age_patches, title='Age Group', loc='upper left', bbox_to_anchor=(1.0, 1))
    ax.add_artist(legend1)
    ax.legend(handles=sex_patches, title='Sex', loc='upper left', bbox_to_anchor=(1.0, 0.6))

    plt.tight_layout()
    out_path = os.path.join(output_folder, "top5_locations_stacked_agegroup_sexpattern.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {out_path}")

