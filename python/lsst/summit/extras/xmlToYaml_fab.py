import requests
import xmltodict
import yaml
import os


# GitHub repository details
owner = 'lsst-ts'
repo = 'ts_xml'
path = 'python/lsst/ts/xml/data/sal_interfaces'
branch = 'develop'
token = 'use_your_own_personal_token'


# GitHub API URL
api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}'


# Fetch the contents of a GitHub directory
def fetch_directory_contents(url):
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


# Recursively fetch all XML files from a directory and its subdirectories
def fetch_all_xml_files(url, xml_files):
    contents = fetch_directory_contents(url)
    for item in contents:
        if item['type'] == 'dir':
            fetch_all_xml_files(item['url'], xml_files)
        elif item['name'].endswith('.xml'):
            xml_files.append(item['download_url'])


# Parse XML content to dictionary
def parse_xml_to_dict(xml_content):
    return xmltodict.parse(xml_content)


# Convert dictionary to YAML
def convert_dict_to_yaml(dictionary):
    return yaml.dump(dictionary, default_flow_style=False)


def main():
    xml_files = []
    all_data = {}
    fetch_all_xml_files(api_url, xml_files)

    for xml_url in xml_files:
        headers = {'Authorization': f'token {token}'}
        xml_response = requests.get(xml_url, headers=headers)
        xml_response.raise_for_status()

        xml_dict = parse_xml_to_dict(xml_response.text)
        # Use the file name as the key to store each XML content
        file_key = xml_url.split('/')[-1].replace('.xml', '')
        all_data[file_key] = xml_dict

    # Save all aggregated data to a single YAML file
    yaml_file_path = os.path.join(os.getcwd(), 'sal_interface.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(all_data, yaml_file, default_flow_style=False)
    print(f"All XML data converted and saved to {yaml_file_path}")


if __name__ == "__main__":
    main()

