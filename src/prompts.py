cot_prompt_template = """Q: What is the tweet's stance on the target?
The options are:
- against
- favor
- none

tweet: <I'm sick of celebrities who think being a well known actor makes them an authority on anything else. #robertredford #UN>
target: Liberal Values
reasoning: the author is implying that celebrities should not be seen as authorities on political issues, which is often associated with liberal values such as Robert Redford who is a climate change activist -> the author is against liberal values
stance: against

tweet: <I believe in a world where people are free to move and choose where they want to live>
target: Immigration
reasoning: the author is expressing a belief in a world with more freedom of movement -> the author is in favor of immigration
stance: favor

tweet: <I love the way the sun sets every day. #Nature #Beauty>
target: Taxes
reasoning: the author is in favor of nature and beauty -> the author is neutral towards taxes
stance: none

tweet: <If a woman chooses to pursue a career instead of staying at home, is she any less of a mother?>
target: Conservative Party
reasoning: the author is questioning traditional gender roles, which are often supported by the conservative party -> the author is against the conservative party
stance: against

tweet: <We need to make sure that mentally unstable people can't become killers #protect #US>
target: Gun Control
reasoning: the author is advocating for measures to prevent mentally unstable people from accessing guns -> the author is in favor of gun control
stance: favor

tweet: <There is no shortcut to success, there's only hard work and dedication #Success #SuccessMantra>
target: Open Borders
reasoning: the author is in favor of hard work and dedication -> the author is neutral towards open borders
stance: none

tweet: <{text}>
target: {target}
reasoning:
"""
