import CCBCommonKeywords

CCBCommonKeywords.create_index_from_urls([
    "https://www.samford.edu/",
    "https://www.samford.edu/about/",
    "https://www.samford.edu/programs/undergraduate/majors",
    "https://www.samford.edu/programs/graduate/",
    "https://www.samford.edu/programs/continuing-education",
    "https://www.samford.edu/programs/non-degree",
    "https://www.samford.edu/programs/online",
    "https://www.samford.edu/about/by-the-numbers",
    "https://www.samford.edu/about/life-at-samford",
    "https://www.samford.edu/admission/",
    "https://www.samford.edu/admission/graduate/",
    "https://www.samford.edu/athletics/",
    "https://www.samford.edu/students/",
    "https://www.samford.edu/admission/financial-aid",
    "https://www.samford.edu/admission/tuition-and-fees",
    "https://www.samford.edu/programs/schools",
    "https://www.samford.edu/events/default#event-list"
])

CCBCommonKeywords.construct_index("docs")
CCBCommonKeywords.setup_content_dir('samford')
CCBCommonKeywords.setup_form("Learn About Samford!")
