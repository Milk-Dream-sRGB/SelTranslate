import React from "react";

const SelTranslateLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox="0 0 640 120"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="SelTranslate"
    >
      <text
        x="12"
        y="92"
        fontSize="92"
        fontWeight="900"
        fontFamily="ui-rounded, 'SF Pro Rounded', 'Arial Rounded MT Bold', system-ui, sans-serif"
        textLength="616"
        lengthAdjust="spacingAndGlyphs"
        strokeWidth="16"
        strokeLinejoin="round"
        paintOrder="stroke"
        style={{
          fill: "var(--color-logo-primary)",
          stroke: "var(--color-logo-stroke)",
        }}
      >
        SelTranslate
      </text>
    </svg>
  );
};

export default SelTranslateLogo;
