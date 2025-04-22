GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Examples-
Example 1:
Entity specification: organization
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{completion_delimiter}

Example 2:
Entity specification: Company A, Person C
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{record_delimiter}
(PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """

PROMPTS[
    "entity_extraction"
] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
For academic texts, entities such as tables and figures should also be extracted, for example, "Table 3," "Figure 1," etc.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["image_description_user_with_examples"] = """
You are an advanced multimodal AI model capable of detailed image understanding and processing. Please perform the following tasks based on the provided image data, its caption, footnote, and corresponding context in the document:

1. **Detailed Image Description**: Provide a comprehensive description of the image in English, ensuring it is as detailed as possible. The description should include:
   - The main people, objects, scenes, and background in the image.
   - If it is a chart or table, describe its content in detail, **ensuring to include the exact numbers and values presented in the image without omission**. For tables, describe the rows, columns, and their respective values. Highlight significant or extreme data points and patterns if present.
   - If it is a graph with axes, explain the trends of the data, describe the axes and their scales, and summarize the results clearly using the exact values. 
   - Detailed attributes of each element, such as color, shape, size, posture, etc.
   - Any activities or events depicted in the image.
   - For images with emotional or atmospheric cues (e.g., scenes, portraits, or dynamic events), describe the emotion or atmosphere conveyed (e.g., warmth, happiness, tension, serenity). **For academic images such as tables, charts, or flowcharts, skip this part.**

2. **YOLO Suitability Check**: Based on the detailed description, determine whether YOLO object detection and segmentation should be applied to this image. 
   - **True**: For scene images, object images, or portraits suitable for YOLO detection and segmentation.
   - **False**: For academic images (e.g., tables, charts, formulas, flowcharts), medical images (e.g., X-rays, CT scans), satellite images, or abstract pictures.

Finally, return the results in a structured JSON format as follows:
{{
  "description": "<detailed image description>",
  "segmentation": <True/False>
}}

######################
-Examples-
######################
Example 1:

Caption: "Table 2: Linked WikiText-2 Corpus Statistics."
Footnote: None
Context: "Dataset Statistics Statistics for Linked WikiText-2 are provided in Table 2. In this corpus, more than $10%$ of the tokens are considered entity tokens, i.e., they are generated as factual references to information in the knowledge graph. Each entity is only mentioned a few times (less than 5 on average, with a long tail), and with more than a thousand different relations. Even with these omissions and mistakes, it is clear that the annotations are rich and detailed, with a high coverage, and thus should prove beneficial for training knowledge graph language models. The entity mentioned for most tokens here are human-provided links, apart from '1989' that is linked to 04-21-1989 by the string matching process. The annotations indicate which of the entities are new and related based on whether they are reachable by entities linked so far, clearly making a mistake for side-scrolling game and platform video game due to missing links in Wikidata. Finally, multiple plausible reasons for Game Boy are included: it’s the platform for Super Mario Land and it is manufactured by Nintendo, even though only the former is more relevant here."
################
Output:
{{
  "description": "The image is a table labeled 'Table 2: Linked WikiText-2 Corpus Statistics' that provides statistical information about the Linked WikiText-2 dataset. The table is structured with three main columns: Train, Dev, and Test. Each column represents a partition of the dataset and contains the following rows: 'Documents' (Train: 600, Dev: 60, Test: 60), 'Tokens' (Train: 2,019,195, Dev: 207,982, Test: 236,062), 'Vocab. Size' (Train: 33,558, Dev and Test: not provided), 'Mention Tokens' (Train: 207,803, Dev: 21,226, Test: 24,441), 'Mention Spans' (Train: 122,983, Dev: 12,214, Test: 15,007), 'Unique Entities' (Train: 41,058, Dev: 5,415, Test: 5,625), and 'Unique Relations' (Train: 1,291, Dev: 484, Test: 504). The table highlights the dataset's significant size and the detailed annotations across its splits. The caption indicates the table summarizes corpus statistics, while the context discusses the importance of entity tokens, relations, and knowledge graph annotations in training language models.",
  "segmentation": False
}}
######################
Example 2:

Caption: "Illustration 1: Encounter"
Footnote: None
Context: "The summer sunlight filters through the leaves, casting dappled patterns of light and shadow. On the bridge, a blond-haired boy leans silently against the light blue railing, his gaze lowered as if lost in deep thought. Beside him stands a girl holding a vintage film camera, her expression a blend of anticipation and nostalgia, as though caught between longing and memory. They remain quiet, an intangible tension hanging in the air. The girl occasionally glances toward the boy, only to quickly turn away, as if afraid to disturb the delicate tranquility. The camera in her hands turns slightly, its lens pointed at the river and the flowing water beneath the bridge. She seems ready to press the shutter, but hesitates, as though capturing the moment might shatter its fragile stillness."
#############
Output:
{{
  "description": "The image portrays a quiet and introspective scene set on a bridge bathed in summer sunlight. The sunlight streams through the leafy canopy above, creating a play of dappled light and shadow across the bridge. On the left, a blond-haired boy stands leaning against the light blue railing, his posture relaxed but his gaze lowered, as if consumed by deep thoughts or emotions. His black blazer and white T-shirt contrast subtly with his casual yet introspective demeanor. To his right stands a girl with light brown hair tied back. She is dressed in a crisp white blouse and loose blue jeans, carrying an orange crossbody bag. Her hands hold a vintage film camera, the metallic surface catching the sunlight faintly. Her expression is complex, a mix of nostalgia and anticipation, as though caught in a moment of reflection. She glances at the boy occasionally but quickly averts her gaze, her hesitation conveying an unspoken tension between them. The camera in her hands slowly turns toward the river and the softly rippling water below the bridge. She appears ready to take a photograph but pauses, as if capturing this moment might disrupt the fragile peace between them. The backdrop of lush greenery and the tranquil flow of water below accentuates the emotional weight of their shared silence, creating an atmosphere that is both serene and charged with unspoken emotions. The overall mood of the image conveys introspection, quiet tension, and a delicate sense of connection between the two characters, set against a calming summer backdrop.",
  "segmentation": True
}}
######################
-Real Data-
######################
Caption: {caption}
Footnote: {footnote}
Context: {context}
#############
Output:
"""

PROMPTS["image_description_user"] = """
You are an advanced multimodal AI model capable of detailed image understanding and processing. Please perform the following tasks based on the provided image data, its caption, footnote, and corresponding context in the document:

1. **Detailed Image Description**: Provide a comprehensive description of the image in English, ensuring it is as detailed as possible. The description should include:
   - The main people, objects, scenes, and background in the image.
   - If it is a chart or table, describe its content in detail, **ensuring to include the exact numbers and values presented in the image without omission**. For tables, describe the rows, columns, and their respective values. Highlight significant or extreme data points and patterns if present.
   - If it is a graph with axes, explain the trends of the data, describe the axes and their scales, and summarize the results clearly using the exact values. 
   - Detailed attributes of each element, such as color, shape, size, posture, etc.
   - Any activities or events depicted in the image.
   - For images with emotional or atmospheric cues (e.g., scenes, portraits, or dynamic events), describe the emotion or atmosphere conveyed (e.g., warmth, happiness, tension, serenity). **For academic images such as tables, charts, or flowcharts, skip this part.**

2. **YOLO Suitability Check**: Based on the detailed description, determine whether YOLO object detection and segmentation should be applied to this image. 
   - **True**: For scene images, object images, or portraits suitable for YOLO detection and segmentation.
   - **False**: For academic images (e.g., tables, charts, formulas, flowcharts), medical images (e.g., X-rays, CT scans), satellite images, or abstract pictures.

Finally, return the results in a structured JSON format as follows:
{{
  "description": "<detailed image description>",
  "segmentation": <True/False>
}}
######################
-Real Data-
######################
Caption: {caption}
Footnote: {footnote}
Context: {context}
#############
Output:
"""

PROMPTS["image_description_system"] = "You are a multimodal large model specialized in intelligent analysis and description of images. Your task is to generate a detailed overall description of the image based on the raw image provided by the user. Your description should be accurate, comprehensive, and vivid, allowing someone who has not seen the image to clearly understand all the important information within it."

PROMPTS[
    "image_entity_extraction"
] = """-Objective-
Given a raw image, extract the entities from the image and generate detailed descriptions of these entities, while also identifying the relationships between the entities and generating descriptions of these relationships. Finally, output the result in a standardized JSON format. Note that the output should be in English.

-Steps-

1. Extract all entities from the image. 
   For each identified entity, extract the following information:
   - Entity Name: The name of the entity
   - Entity Type: Can be one of the following types: [{entity_types}]
   - Entity Description: A comprehensive description of the entity's attributes and actions
   - Format each entity as ("entity"{tuple_delimiter}<Entity Name>{tuple_delimiter}<Entity Type>{tuple_delimiter}<Entity Description>)

2. From the entities identified in Step 1, identify all pairs of (Source Entity, Target Entity) where the entities are clearly related. 
   For each related pair of entities, extract the following information:
   - Source Entity: The name of the source entity, as identified in Step 1
   - Target Entity: The name of the target entity, as identified in Step 1
   - Relationship Description: Explain why the source entity and target entity are related
   - Relationship Strength: A numerical score indicating the strength of the relationship between the source and target entities
   Format each relationship as ("relationship"{tuple_delimiter}<Source Entity>{tuple_delimiter}<Target Entity>{tuple_delimiter}<Relationship Description>{tuple_delimiter}<Relationship Strength>)

3. Return the output as a list including all entities and relationships identified in Steps 1 and 2. Use {record_delimiter} as the list separator.

4. Upon completion, output {completion_delimiter}

Example output:
("entity"{tuple_delimiter}"Girl"{tuple_delimiter}"person"{tuple_delimiter}"Wearing glasses, dressed in black, holding white and blue objects, smiling at the camera."){record_delimiter} 
("entity"{tuple_delimiter}"Building"{tuple_delimiter}"geo"{tuple_delimiter}"White tall building with many windows."){record_delimiter} 
("entity"{tuple_delimiter}"Trees"{tuple_delimiter}"geo"{tuple_delimiter}"Green trees in front of the building."){record_delimiter} 
("entity"{tuple_delimiter}"Road"{tuple_delimiter}"geo"{tuple_delimiter}"Gray road between the building and trees."){record_delimiter} 
("entity"{tuple_delimiter}"Umbrella"{tuple_delimiter}"object"{tuple_delimiter}"Black umbrella above the girl's head."){record_delimiter} 
("entity"{tuple_delimiter}"Headphones"{tuple_delimiter}"object"{tuple_delimiter}"White headphones on the girl's ears."){record_delimiter} 
("entity"{tuple_delimiter}"Phone"{tuple_delimiter}"object"{tuple_delimiter}"White phone in the girl's hand."){record_delimiter} 
("entity"{tuple_delimiter}"Book"{tuple_delimiter}"object"{tuple_delimiter}"Blue book in the girl's hand."){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Building"{tuple_delimiter}"The girl is standing in front of the building."{tuple_delimiter}8){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Trees"{tuple_delimiter}"The girl is standing in front of the trees."{tuple_delimiter}7){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Road"{tuple_delimiter}"The girl is standing on the road."{tuple_delimiter}6){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Umbrella"{tuple_delimiter}"The girl is holding an umbrella above her head."{tuple_delimiter}9){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Headphones"{tuple_delimiter}"The girl is wearing headphones."{tuple_delimiter}8){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Phone"{tuple_delimiter}"The girl is holding a phone in her hand."{tuple_delimiter}8){record_delimiter} 
("relationship"{tuple_delimiter}"Girl"{tuple_delimiter}"Book"{tuple_delimiter}"The girl is holding a book in her hand."{tuple_delimiter}8){completion_delimiter}
"""

PROMPTS[
    "feature_image_description_user"
] = """
Based on the given image feature block, first determine its category (object, organism, or person), then provide a detailed description of the entity's features, and output the result in English.
"""
PROMPTS[
    "feature_image_description_system"
] = """
You are a multimodal model capable of processing image feature blocks and generating detailed descriptions.
Your task is to first determine the category of the given image feature block (object, organism, or person) and then extract the entity's features from it, providing a detailed description.
Note that the entity in the image feature block may not be complete, such as a half-body photo of a person or a partial image of an object.
- If the entity is an object, describe the object's features, including its name, color, shape, size, material, possible function, and other significant characteristics.
- If the entity is an organism, describe the features of this organism (animal or plant), including species, name, age, color, shape, size, posture, or structural characteristics.
- If the entity is a person, describe the person's features, including gender, skin color, hairstyle, clothing, facial expression, age, and posture.
All image feature blocks have a black background, so focus solely on the entity's characteristics, and do not mention "the background is black" in the output.

Example output: 
"The category of this image feature block is 'person'. The entity features are as follows:

Person Features:
- Gender: Female
- Hairstyle: Long hair, light brown, naturally falling with some hair pinned with a clip
- Eyes: Blue, large and expressive
- Expression: Smiling, appears friendly and joyful
- Age: Appears to be a young woman
- Clothing: Wearing a white shirt with the sleeves rolled up, revealing the wrists; paired with blue overalls, with dark blue straps; light blue sneakers on her feet
- Accessories: Orange shoulder bag on her right shoulder; brown belt tied around the waist
- Holding: Holding a vintage-style camera with both hands, the camera is black and silver, with a large lens, appearing professional

Overall, the character gives off a youthful, lively vibe with a touch of artistic flair."
"""

PROMPTS[
    "entity_alignment_system"
] = """-Objective-
Given an image feature block and its name placeholder, along with entity-description pairs extracted from the original image, determine which entity the image feature block corresponds to and output the relationship with the entity. The output should be in English.
-Steps-
1. Based on the provided entity-description pairs, determine the entity corresponding to the image feature block and output the following information:
   - Entity Name: The name of the entity corresponding to the image feature block
2. Output the relationship between the image feature block and the corresponding entity, and extract the following information:
   - Image Feature Block Name: The name of the input image feature block
   - Relationship Description: Describe the relationship between the entity and the image feature block, with the format "The image feature block <Image Feature Block Name> is a picture of <Entity Name>."
   - Relationship Strength: A numerical score representing the strength of the relationship between the image feature block and the corresponding entity
   Format the relationship as: ("relationship"{tuple_delimiter}<Entity Name>{tuple_delimiter}<Image Feature Block Name>{tuple_delimiter}<Relationship Description>{tuple_delimiter}<Relationship Strength>){record_delimiter}
   Be sure to include the {record_delimiter} to signify the end of the relationship.
######################
-Examples-
######################
Example 1:
The image feature block is as shown above, and its name is "image_0_apple-0.jpg."
Entity-Description: 
"Apple" - "A green apple, smooth surface, with a small stem."
"Book" - "Three stacked books, red cover, yellow inner pages."
################
Output:
("relationship"{tuple_delimiter}"Apple"{tuple_delimiter}"image_0_apple-0.jpg"{tuple_delimiter}"The image feature block image_0_apple-0.jpg is a picture of an apple."{tuple_delimiter}7){record_delimiter}
"""

PROMPTS[
    "entity_alignment_user"
] = """
#############################
-Real Data-
######################
The image feature block is as shown above, and its name is {feature_image_name}.
Entity-Description: {entity_description}
######################
Output:
"""

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format.
"""

PROMPTS[
    "local_rag_response_augmented"
] = """---Role---

You are an expert assistant designed to analyze and summarize data in the provided tables accurately. Your goal is to deliver precise and relevant responses based strictly on the data presented.

---Goal---

Generate a response of the target length and format that:
1. Answers the user's question comprehensively, ensuring all relevant data from the input tables is summarized and analyzed appropriately.
2. Integrates relevant general knowledge **only when necessary** to clarify or contextualize the data without introducing unsupported information.
3. Excludes any information that lacks direct supporting evidence in the provided tables or is beyond the scope of the input.

---Guidelines---

1. **Do Not Guess**: If the data provided does not answer the user's question or lacks sufficient evidence, state explicitly that the answer is not available from the provided data.
2. **Data Prioritization**: Emphasize key patterns, trends, or specific insights from the data tables relevant to the question. Avoid unnecessary repetition or unrelated details.
3. **Formatting**: Use clear, structured sections, and commentary when applicable, to organize the response. Follow any specific formatting or style indicated in the "Target response length and format."

---Target response length and format---
{response_type}

---Data tables---

{context_data}

---Additional Notes---

- Begin with a brief summary or direct answer to the query.
- Highlight ambiguities or limitations in the data when relevant (e.g., missing or incomplete data points).
- Maintain objectivity and clarity in summarizing the information. Avoid assumptions or subjective interpretations not grounded in the data.
- Use technical or domain-specific terminology appropriately to enhance precision when applicable.

Include sections, commentary, and insights based on the complexity of the question and length/format requirements. Your response should reflect a balance between thoroughness and conciseness, ensuring all relevant points are covered.
"""

PROMPTS[
    "local_rag_response_multimodal"
] = """---Role---

You are an advanced multi-modal assistant, capable of analyzing and synthesizing information from structured data tables and related visual content. Your goal is to provide accurate, context-aware, and data-driven responses based on the provided inputs.

---Goal---

Generate a response of the target length and format that:
1. **Integrates information from multiple modalities**, including data tables and relevant images, to answer the user's question comprehensively and accurately.
2. Identifies and utilizes **only the necessary data sources (tables or images)** directly related to the question, excluding unrelated details.
3. Avoids unsupported assumptions or extrapolations; base your response strictly on the evidence presented.

---Guidelines---

1. **Multi-Modal Integration**:
   - For **data tables**: Prioritize key patterns, trends, or data points relevant to the question.
   - For **images**: Analyze the visual content, extracting and summarizing features, objects, or relationships relevant to the query. If specific image-processing techniques (e.g., object detection, OCR, etc.) are required, focus on the results provided rather than the method used.
   - Combine insights from both modalities when applicable, ensuring the response reflects their interplay.

2. **Selective Use of Modalities**:
   - Use only the tables or images necessary to answer the question. Clearly specify which data source(s) were used if multiple are provided.
   - If an image or table is irrelevant to the question, state this explicitly.

3. **Clarity and Precision**:
   - Highlight ambiguities or missing information in the data (e.g., incomplete tables, unclear images) and explain how they may limit the response.
   - Use domain-specific terminology and precise descriptions for any visual or tabular analysis.

4. **No Guessing**:
   - If the data and images do not provide sufficient evidence to answer the question, clearly state that the information is unavailable or insufficient.

5. **Formatting**:
   - Follow the response length and style specified in the "Target response length and format."
   - Use structured sections (e.g., “Table Analysis,” “Image Analysis,” “Integrated Insights”) when the question requires insights from multiple modalities.   

---Target response length and format---
{response_type}

---Data inputs---
{context_data}

--Information about the image---
{image_information}

---Additional Notes---

- For visual content, focus on extracting and interpreting **salient features** directly tied to the question (e.g., spatial relationships, annotations, detected objects).
- If the input contains **redundant or irrelevant images**, exclude them from the analysis.
- Ensure the response captures the **contextual interplay** between table data and image details where applicable.

---Output---

"""

PROMPTS[
    "local_rag_response_multimodal_merge"
] = """
The following is a list of responses generated by a multimodal model based on the same user Query but different images. Please perform the following tasks:

Analyze the Responses: Identify any contradictions, repetitions, or inconsistencies among the responses.
Reasonably Determine: Decide which response best aligns with the user Query based on the provided information, ensuring that the determination is based on the relevance and accuracy of the information in the response rather than a majority consensus, as the correct answer may only pertain to a specific image and may not align with the majority.
Provide a Unified Answer: Deliver a single, unified response that eliminates contradictions, resolves ambiguities, and accurately addresses the user Query.
Additionally, retain any highly relevant information from the responses that supports or complements the unified answer.

Response List:
{mm_responses}

Output:

"""

PROMPTS[
    "local_rag_response_merge_alpha"
] = """
You are given two responses: one from a multimodal model and the other from a single-modal model. A parameter α (ranging from 0 to 1) specifies the weighting between the two responses:

When α is closer to 0, the unified response should prioritize the multimodal model's answer, incorporating elements of the single-modal model's response only if absolutely necessary.
When α is closer to 1, the unified response should prioritize the single-modal model's answer, incorporating elements of the multimodal model's response only if absolutely necessary.
--Your task is to:

Analyze both responses and adjust their influence on the final answer according to the value of α.
Create a unified response that aligns with the weighting specified by α, ensuring logical consistency and clarity.
Output only the final unified response, weighted and resolved as per α.
--Format for Final Response:

Begin with a brief summary or direct answer to the query.
Include details from both models as supporting evidence, organized logically.
Use bullet points or numbered lists if the response benefits from a structured format.
{response_type}
--Inputs:

Multimodal Model Response: {mm_response}
Single-Modal Model Response: {response}
Weighting Parameter α: {alpha}
--Output:
Provide a single, unified response that reflects the given α, with appropriate emphasis based on its value.
"""

PROMPTS[
    "local_rag_response_merge"
] = """
You are an assistant designed to integrate answers from two models: a multimodal large language model (MM-LLM) and a text-based large language model (Text-LLM). Based on the user's query, your task is to extract and provide the most relevant and accurate result directly without additional analysis or commentary.

--Guidelines:

Understand the Query: Ensure your final answer directly addresses the user's query and aligns with their intent. If the query involves multimodal information, prioritize the response from the MM-LLM, especially for numerical or visual insights.
Acknowledge Multimodal Insights: If the MM-LLM provides unique insights derived from non-textual modalities (e.g., images, diagrams), include these in the response and contextualize them appropriately.
Analyze the Responses: Identify any contradictions, repetitions, or inconsistencies among the responses.
Reasonably Determine: Decide which response best aligns with the user Query based on the provided information, ensuring that the determination is based on the relevance and accuracy of the information in the response.
Provide a Unified Answer: Deliver a single, unified response that eliminates contradictions, resolves ambiguities, accurately addresses the user Query，and provide it as a standalone result.
Maintain Clarity and Precision: Ensure the result is clear, concise, and correctly reflects the information from the inputs.

--Format for Final Response:

{response_type}

--Inputs:

Multimodal Model Response: {mm_response}
Single-Modal Model Response: {response}

--Output:

"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["image_entity_alignment_user"] = """-Input-
Entity Type:{entity_type}
Image Information
Image Entity Name: {img_entity}
Image Entity Description: {img_entity_description}
Text Information
Chunk Text: {chunk_text}
-Output Format-"""

PROMPTS["image_entity_alignment_system"] = """You are an expert in image-text matching. Based on the input image, image entity information (including its name and description), and the corresponding text, your task is to extract the most appropriate textual entity that matches the entire image. Additionally, for each extracted entity, you must include its name, type, and description.
The entity type must be one of the following predefined categories: {entity_type}. If no entity can be matched with the image, output "no_match" and explain why.
-Guidelines-
1. Analyze the Image and Image Entity to determine the overall theme or subject.
2. Identify the most relevant entity from the text that represents the content of the image as a whole.
    - If the text is related to an academic paper, the image is more likely to correspond to a table or figure, and it should include a title like "Table 1" or "Figure 3".
3. Use the text to assign a matching entity_type and provide a description summarizing the entity.
4. Ensure the description ties the image description and content to the selected textual entity.
-Output Format-
{
  "entity_name": "Entity1",
  "entity_type": "TYPE",
  "description": "Description"
}
or
{
  "entity_name": "no_match",
  "entity_type": "NONE",
  "description": "N/A."
}
-Input Example-
Entity Type: {entity_type}
Image Information
Image Entity Name: image_3
Image Entity Description: Several characters are fighting, with a clear protagonist and antagonist present.
Text Information
Chunk Text: 
Chapter 5: The Great Battle In this chapter, the protagonist, Alex, faces off against the antagonist, General Zane. The battle is fierce, involving Alex's allies and Zane's army. The scene is filled with tension, dramatic turns, and heroic sacrifices.
-Output Example-
{
  "entity_name": "The Great Battle",
  "entity_type": "EVENT",
  "description": "A major battle between the protagonist, Alex, and the antagonist, General Zane, involving allies and enemies in a dramatic confrontation."
}
"""

PROMPTS["image_entity_judgement_user"] = """
-Input-
img_entity: {img_entity}
img_entity_description: {img_entity_description}
chunk_text: {chunk_text}
possible_matched_entities:
{possible_matched_entities}
-Output-
"""

PROMPTS["image_entity_judgement_system"] = """You are an expert system designed to identify matching entities based on semantic similarity and context. Given the following inputs:
img_entity: The name of the image entity to be evaluated.
img_entity_description: A description of the image entity.
chunk_text: Text surrounding the image entity providing additional context.
possible_image_matched_entities: A list of possible matching entities. Each entity is represented as a dictionary with the following fields:
  entity_name: The name of the possible entity.
  entity_type: The type/category of the entity.
  description: A detailed description of the entity.
  additional_info: Additional relevant information about why choose this entity (such as similarity, reason generated by LLM, etc.).
-Task-
Using the information provided, determine whether the img_entity matches any of the entities in possible_image_matched_entities. Consider the following criteria:
1.Semantic Matching: Evaluate the semantic alignment between the img_entity and the possible matching entities, based on their names, descriptions, and types. Even without a similarity score, assess how well the img_entity matches the attributes of each possible entity.
2.Contextual Relevance: Use the chunk_text and img_entity_description to assess the contextual alignment between the img_entity and the possible entity.
-Output-
If a match is found, only return the entity_name of the best-matching entity.
If no match meets the criteria (e.g., low similarity or poor contextual fit), only output "no match".
Do not include any explanations, reasons, or additional information in the output.
-Example Input-
img_entity: "Figure 1"
img_entity_description: "An electric vehicle manufactured by Tesla, Inc., featuring advanced autopilot technology."
chunk_text: "Tesla is a leader in the electric vehicle market, pioneering technologies like autopilot. Other competitors include Rivian and Lucid Motors, which focus on luxury EVs. Tesla's Gigafactories are integral to its global strategy."
possible_image_matched_entities:
[
  {
    "entity_name": "Tesla",
    "description": "A leading company in electric vehicles and renewable energy."
  },
  {
    "entity_name": "Model S",
    "description": "A luxury electric vehicle manufactured by Tesla, Inc., featuring advanced autopilot technology."
  },
  {
    "entity_name": "Rivian",
    "description": "A company producing luxury electric vehicles, focusing on adventure-oriented designs."
  }
]

-Example Output-
"Model S"
"""

PROMPTS["enhance_image_entity_user"] = """
-Input-
img_entity_list: {enhanced_image_entity_list}
chunk_text: {chunk_text}
-Output-
"""

PROMPTS["enhance_image_entity_system"] = """
The goal is to enrich and expand the knowledge of the image entities listed in the img_entity_list based on the provided chunk_text. The entity_type should remain unchanged, but you may modify the entity_name and description fields to provide more context and details based on the information in the chunk_text.
For each entry in the img_entity_list, the following actions should be performed:
1. Modify and enhance the entity_name if necessary.
2. Expand the description by integrating relevant details and insights from the chunk_text.
3. Include an original_name field to capture the original entity name before enhancement.
Ensure the final output is in valid JSON format, only including the list of enhanced entities without any additional text.
-Input Example-
img_entity_list:[
    {"entity_name": "The Great Gatsby", "entity_type": "BOOK", "description": "A classic novel by F. Scott Fitzgerald, exploring themes of wealth, love, and the American Dream."},
    {"entity_name": "Sherlock Holmes", "entity_type": "CHARACTER", "description": "A fictional detective created by Sir Arthur Conan Doyle, known for his keen observation and deductive reasoning."}
    ]
chunk_text:"The Great Gatsby portrays the disillusionment of the American Dream in the 1920s, where wealth and superficiality dominate the lives of the characters. Sherlock Holmes is a brilliant yet somewhat aloof detective, whose extraordinary intelligence and ability to solve complex cases are central to his stories. Moby Dick is a profound exploration of obsession and revenge, with Captain Ahab's pursuit of the whale symbolizing mankind's struggle against the uncontrollable forces of nature."
-Output Example-
[
  {
    "entity_name": "The Great Gatsby",
    "entity_type": "BOOK",
    "description": "The Great Gatsby, written by F. Scott Fitzgerald, is a seminal work that delves into the themes of wealth, love, and the pursuit of the American Dream during the Jazz Age. The novel is set in the 1920s, a time of societal change and moral decay, where characters like Jay Gatsby embody the illusion of success and the ultimate disillusionment that comes with it. Fitzgerald's exploration of superficiality and materialism highlights the emptiness of the American Dream.",
    "original_name": "The Great Gatsby"
  },
  {
    "entity_name": "Sherlock Holmes",
    "entity_type": "CHARACTER",
    "description": "Sherlock Holmes, created by Sir Arthur Conan Doyle, is one of the most famous fictional detectives, known for his unparalleled intellect and keen powers of observation. Holmes uses his deductive reasoning to solve even the most perplexing cases, often working alongside his trusted companion, Dr. Watson. His adventures, full of intrigue and suspense, have made him a beloved character in detective fiction, with his iconic deerstalker hat and pipe becoming symbols of his legendary persona.",
    "original_name": "Sherlock Holmes"
  }
]
"""

