ENTITY_EXTRACTION_EXAMPLE_STORY = """Example 1
Input:
{
  "entity_types": [__ENTITY_TYPES__],
  "input_text": "while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.\\n\\nThen Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. \\"If this tech can be understood...\\" Taylor said, their voice quieter, \\"It could change the game for us. For all of us.\\"\\n\\nThe underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.\\n\\nIt was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths"
}

Output:
{
  "extraction_output": [
    {
      "source": {
        "name": "Alex",
        "type": "Person",
        "description": "Alex experiences frustration and closely observes the shifting dynamics among the group."
      },
      "target": {
        "name": "Taylor",
        "type": "Person",
        "description": "Taylor initially projects authoritarian certainty and later shows reluctant respect toward the device and Jordan."
      },
      "keywords": ["power dynamics", "observation"],
      "description": "Alex observes Taylor's authoritarian posture and notices Taylor's change in attitude."
    },
    {
      "source": {
        "name": "Alex",
        "type": "Person",
        "description": "Alex experiences frustration and closely observes the shifting dynamics among the group."
      },
      "target": {
        "name": "Jordan",
        "type": "Person",
        "description": "Jordan shares a commitment to discovery and directly engages with Taylor around the device."
      },
      "keywords": ["shared commitment", "discovery"],
      "description": "Alex and Jordan share a commitment to discovery."
    },
    {
      "source": {
        "name": "Jordan",
        "type": "Person",
        "description": "Jordan shares a commitment to discovery and directly engages with Taylor around the device."
      },
      "target": {
        "name": "Cruz",
        "type": "Person",
        "description": "Cruz is associated with a narrowing vision of control and order."
      },
      "keywords": ["ideological conflict", "rebellion"],
      "description": "Jordan's commitment to discovery stands in rebellion against Cruz's vision of control and order."
    },
    {
      "source": {
        "name": "Taylor",
        "type": "Person",
        "description": "Taylor initially projects authoritarian certainty and later shows reluctant respect toward the device and Jordan."
      },
      "target": {
        "name": "Jordan",
        "type": "Person",
        "description": "Jordan shares a commitment to discovery and directly engages with Taylor around the device."
      },
      "keywords": ["conflict", "uneasy truce"],
      "description": "Taylor and Jordan move from tension toward an uneasy truce while considering the device."
    },
    {
      "source": {
        "name": "Taylor",
        "type": "Person",
        "description": "Taylor initially projects authoritarian certainty and later shows reluctant respect toward the device and Jordan."
      },
      "target": {
        "name": "Device",
        "type": "Artifact",
        "description": "The device is a technology that the group regards as potentially transformative."
      },
      "keywords": ["reverence", "technology"],
      "description": "Taylor treats the device with reverence and recognizes its possible significance."
    }
  ]
}
"""

ENTITY_EXTRACTION_EXAMPLE_MARKETS = """Example 2
Input:
{
  "entity_types": [__ENTITY_TYPES__],
  "input_text": "Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.\\n\\nAmong the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.\\n\\nMeanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.\\n\\nFinancial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability."
}

Output:
{
  "extraction_output": [
    {
      "source": {
        "name": "Global Tech Index",
        "type": "Data",
        "description": "The Global Tech Index drops by 3.4% in midday trading during a broader market downturn."
      },
      "target": {
        "name": "Market Selloff",
        "type": "Event",
        "description": "The market selloff is a sharp downturn in stock markets driven by investor concerns."
      },
      "keywords": ["market performance", "decline"],
      "description": "The Global Tech Index decline is part of the broader market selloff."
    },
    {
      "source": {
        "name": "Nexon Technologies",
        "type": "Organization",
        "description": "Nexon Technologies reports lower-than-expected quarterly earnings and its stock falls by 7.8%."
      },
      "target": {
        "name": "Global Tech Index",
        "type": "Data",
        "description": "The Global Tech Index drops by 3.4% in midday trading during a broader market downturn."
      },
      "keywords": ["index movement", "stock decline"],
      "description": "Nexon Technologies' decline contributes to weakness in the Global Tech Index."
    },
    {
      "source": {
        "name": "Omega Energy",
        "type": "Organization",
        "description": "Omega Energy gains 2.1% as rising oil prices support its stock."
      },
      "target": {
        "name": "Crude Oil",
        "type": "NaturalObject",
        "description": "Crude oil prices rise to $87.60 per barrel due to supply constraints and strong demand."
      },
      "keywords": ["price support", "energy markets"],
      "description": "Omega Energy's gain is driven by rising crude oil prices."
    },
    {
      "source": {
        "name": "Gold Futures",
        "type": "Artifact",
        "description": "Gold futures rise by 1.5% to $2,080 per ounce as investors seek safe-haven assets."
      },
      "target": {
        "name": "Market Selloff",
        "type": "Event",
        "description": "The market selloff is a sharp downturn in stock markets driven by investor concerns."
      },
      "keywords": ["safe-haven demand", "market reaction"],
      "description": "Gold futures rise as investors react to the market selloff by seeking safe-haven assets."
    },
    {
      "source": {
        "name": "Federal Reserve Policy Announcement",
        "type": "Event",
        "description": "The upcoming Federal Reserve policy announcement is expected to influence investor confidence and market stability."
      },
      "target": {
        "name": "Market Selloff",
        "type": "Event",
        "description": "The market selloff is a sharp downturn in stock markets driven by investor concerns."
      },
      "keywords": ["rate hikes", "market uncertainty"],
      "description": "Speculation around the Federal Reserve policy announcement contributes to the market selloff."
    }
  ]
}
"""

ENTITY_EXTRACTION_EXAMPLE_ATHLETICS = """Example 3
Input:
{
  "entity_types": [__ENTITY_TYPES__],
  "input_text": "At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes."
}

Output:
{
  "extraction_output": [
    {
      "source": {
        "name": "World Athletics Championship",
        "type": "Event",
        "description": "The World Athletics Championship is the event where Noah Carter breaks the 100m sprint record."
      },
      "target": {
        "name": "Tokyo",
        "type": "Location",
        "description": "Tokyo is the city hosting the World Athletics Championship."
      },
      "keywords": ["event location", "hosting"],
      "description": "The World Athletics Championship takes place in Tokyo."
    },
    {
      "source": {
        "name": "Noah Carter",
        "type": "Person",
        "description": "Noah Carter is the athlete who breaks the 100m sprint record at the championship."
      },
      "target": {
        "name": "100m Sprint Record",
        "type": "Data",
        "description": "The 100m sprint record is the performance mark broken by Noah Carter."
      },
      "keywords": ["record-breaking", "athletic achievement"],
      "description": "Noah Carter breaks the 100m sprint record."
    },
    {
      "source": {
        "name": "Noah Carter",
        "type": "Person",
        "description": "Noah Carter is the athlete who breaks the 100m sprint record at the championship."
      },
      "target": {
        "name": "Carbon-Fiber Spikes",
        "type": "Artifact",
        "description": "Carbon-fiber spikes are the cutting-edge shoes Noah Carter uses during the record-breaking performance."
      },
      "keywords": ["equipment", "performance"],
      "description": "Noah Carter uses carbon-fiber spikes during the record-breaking sprint."
    },
    {
      "source": {
        "name": "Noah Carter",
        "type": "Person",
        "description": "Noah Carter is the athlete who breaks the 100m sprint record at the championship."
      },
      "target": {
        "name": "World Athletics Championship",
        "type": "Event",
        "description": "The World Athletics Championship is the event where Noah Carter breaks the 100m sprint record."
      },
      "keywords": ["participation", "competition"],
      "description": "Noah Carter competes in the World Athletics Championship."
    }
  ]
}
"""

ENTITY_EXTRACTION_EXAMPLES = [
    ENTITY_EXTRACTION_EXAMPLE_STORY,
    ENTITY_EXTRACTION_EXAMPLE_MARKETS,
    ENTITY_EXTRACTION_EXAMPLE_ATHLETICS,
]


def render_entity_extraction_examples(
    *,
    tuple_delimiter: str,
    completion_delimiter: str,
    entity_types: str,
) -> str:
    del tuple_delimiter, completion_delimiter
    return "\n\n".join(ENTITY_EXTRACTION_EXAMPLES).replace(
        "__ENTITY_TYPES__", entity_types
    )
