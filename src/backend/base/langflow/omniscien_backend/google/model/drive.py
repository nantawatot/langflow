from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field


class Status(BaseModel):
    code: int | None = Field(None, description="The status code, which should be an enum value of google.rpc.Code.")
    message: str | None = Field(
        None,
        description="A developer-facing error message, which should be in English. Any user-facing error message should be localized and sent in the google.rpc.Status.details field, or localized by the client.",
    )
    details: list[dict[str, Any]] | None = Field(
        None,
        description="A list of messages that carry the error details. There is a common set of message types for APIs to use.",
    )


class StorageQuota(BaseModel):
    limit: int | None = Field(
        None,
        description="The usage limit, if applicable. This will not be present if the user has unlimited storage. For users that are part of an organization with pooled storage, this is the limit for the organization, rather than the individual user.",
    )
    usageInDrive: int | None = Field(None, description="The usage by all files in Google Drive.")
    usageInDriveTrash: int | None = Field(None, description="The usage by trashed files in Google Drive.")
    usage: int | None = Field(
        None,
        description="The total usage across all services. For users that are part of an organization with pooled storage, this is the usage across all services for the organization, rather than the individual user.",
    )


class DriveTheme(BaseModel):
    id: str | None = Field(None, description="The ID of the theme.")
    backgroundImageLink: str | None = Field(None, description="A link to this theme's background image.")
    colorRgb: str | None = Field(None, description="The color of this theme as an RGB hex string.")


class TeamDriveTheme(BaseModel):
    id: str | None = Field(None, description="Deprecated: Use `driveThemes/id` instead.")
    backgroundImageLink: str | None = Field(
        None, description="Deprecated: Use `driveThemes/backgroundImageLink` instead."
    )
    colorRgb: str | None = Field(None, description="Deprecated: Use `driveThemes/colorRgb` instead.")


class User(BaseModel):
    displayName: str | None = Field(None, description="Output only. A plain text displayable name for this user.")
    kind: str | None = Field(
        "drive#user",
        description="Output only. Identifies what kind of resource this is. Value: the fixed string `drive#user`.",
    )
    me: bool | None = Field(None, description="Output only. Whether this user is the requesting user.")
    permissionId: str | None = Field(None, description="Output only. The user's ID as visible in Permission resources.")
    emailAddress: str | None = Field(
        None,
        description="Output only. The email address of the user. This may not be present in certain contexts if the user has not made their email address visible to the requester.",
    )
    photoLink: str | None = Field(None, description="Output only. A link to the user's profile photo, if available.")


class AppIcons(BaseModel):
    size: int | None = Field(None, description="Size of the icon. Represented as the maximum of the width and height.")
    category: str | None = Field(
        None,
        description="Category of the icon. Allowed values are: * `application` - The icon for the application. * `document` - The icon for a file associated with the app. * `documentShared` - The icon for a shared file associated with the app.",
    )
    iconUrl: str | None = Field(None, description="URL for the icon.")


class StartPageToken(BaseModel):
    startPageToken: str | None = Field(
        None, description="The starting page token for listing future changes. The page token doesn't expire."
    )
    kind: str | None = Field(
        "drive#startPageToken",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#startPageToken"`.',
    )


class Thumbnail(BaseModel):
    image: str | None = Field(
        None,
        description="The thumbnail data encoded with URL-safe Base64 ([RFC 4648 section 5](https://datatracker.ietf.org/doc/html/rfc4648#section-5)).",
    )
    mimeType: str | None = Field(None, description="The MIME type of the thumbnail.")


class ContentHints(BaseModel):
    indexableText: str | None = Field(
        None,
        description="Text to be indexed for the file to improve fullText queries. This is limited to 128 KB in length and may contain HTML elements.",
    )
    thumbnail: Thumbnail | None = Field(
        None,
        description="A thumbnail for the file. This will only be used if Google Drive cannot generate a standard thumbnail.",
    )


class Capabilities(BaseModel):
    canChangeViewersCanCopyContent: bool | None = Field(None, description="Deprecated: Output only.")
    canMoveChildrenOutOfDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can move children of this folder outside of the shared drive. This is `false` when the item isn't a folder. Only populated for items in shared drives.",
    )
    canReadDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can read the shared drive to which this file belongs. Only populated for items in shared drives.",
    )
    canEdit: bool | None = Field(
        None,
        description="Output only. Whether the current user can edit this file. Other factors may limit the type of changes a user can make to a file. For example, see `canChangeCopyRequiresWriterPermission` or `canModifyContent`.",
    )
    canCopy: bool | None = Field(
        None,
        description="Output only. Whether the current user can copy this file. For an item in a shared drive, whether the current user can copy non-folder descendants of this item, or this item if it's not a folder.",
    )
    canComment: bool | None = Field(None, description="Output only. Whether the current user can comment on this file.")
    canAddChildren: bool | None = Field(
        None,
        description="Output only. Whether the current user can add children to this folder. This is always `false` when the item isn't a folder.",
    )
    canDelete: bool | None = Field(None, description="Output only. Whether the current user can delete this file.")
    canDownload: bool | None = Field(None, description="Output only. Whether the current user can download this file.")
    canListChildren: bool | None = Field(
        None,
        description="Output only. Whether the current user can list the children of this folder. This is always `false` when the item isn't a folder.",
    )
    canRemoveChildren: bool | None = Field(
        None,
        description="Output only. Whether the current user can remove children from this folder. This is always `false` when the item isn't a folder. For a folder in a shared drive, use `canDeleteChildren` or `canTrashChildren` instead.",
    )
    canRename: bool | None = Field(None, description="Output only. Whether the current user can rename this file.")
    canTrash: bool | None = Field(
        None, description="Output only. Whether the current user can move this file to trash."
    )
    canReadRevisions: bool | None = Field(
        None,
        description="Output only. Whether the current user can read the revisions resource of this file. For a shared drive item, whether revisions of non-folder descendants of this item, or this item if it's not a folder, can be read.",
    )
    canReadTeamDrive: bool | None = Field(None, description="Deprecated: Output only. Use `canReadDrive` instead.")
    canMoveTeamDriveItem: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveItemWithinDrive` or `canMoveItemOutOfDrive` instead."
    )
    canChangeCopyRequiresWriterPermission: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `copyRequiresWriterPermission` restriction of this file.",
    )
    canMoveItemIntoTeamDrive: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveItemOutOfDrive` instead."
    )
    canUntrash: bool | None = Field(
        None, description="Output only. Whether the current user can restore this file from trash."
    )
    canModifyContent: bool | None = Field(
        None, description="Output only. Whether the current user can modify the content of this file."
    )
    canMoveItemWithinTeamDrive: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveItemWithinDrive` instead."
    )
    canMoveItemOutOfTeamDrive: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveItemOutOfDrive` instead."
    )
    canDeleteChildren: bool | None = Field(
        None,
        description="Output only. Whether the current user can delete children of this folder. This is `false` when the item isn't a folder. Only populated for items in shared drives.",
    )
    canMoveChildrenOutOfTeamDrive: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveChildrenOutOfDrive` instead."
    )
    canMoveChildrenWithinTeamDrive: bool | None = Field(
        None, description="Deprecated: Output only. Use `canMoveChildrenWithinDrive` instead."
    )
    canTrashChildren: bool | None = Field(
        None,
        description="Output only. Whether the current user can trash children of this folder. This is `false` when the item isn't a folder. Only populated for items in shared drives.",
    )
    canMoveItemOutOfDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can move this item outside of this drive by changing its parent. Note that a request to change the parent of the item may still fail depending on the new parent that's being added.",
    )
    canAddMyDriveParent: bool | None = Field(
        None,
        description="Output only. Whether the current user can add a parent for the item without removing an existing parent in the same request. Not populated for shared drive files.",
    )
    canRemoveMyDriveParent: bool | None = Field(
        None,
        description="Output only. Whether the current user can remove a parent from the item without adding another parent in the same request. Not populated for shared drive files.",
    )
    canMoveItemWithinDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can move this item within this drive. Note that a request to change the parent of the item may still fail depending on the new parent that's being added and the parent that is being removed.",
    )
    canShare: bool | None = Field(
        None, description="Output only. Whether the current user can modify the sharing settings for this file."
    )
    canMoveChildrenWithinDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can move children of this folder within this drive. This is `false` when the item isn't a folder. Note that a request to move the child may still fail depending on the current user's access to the child and to the destination folder.",
    )
    canModifyContentRestriction: bool | None = Field(
        None,
        description="Deprecated: Output only. Use one of `canModifyEditorContentRestriction`, `canModifyOwnerContentRestriction`, or `canRemoveContentRestriction`.",
    )
    canAddFolderFromAnotherDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can add a folder from another drive (different shared drive or My Drive) to this folder. This is `false` when the item isn't a folder. Only populated for items in shared drives.",
    )
    canChangeSecurityUpdateEnabled: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `securityUpdateEnabled` field on link share metadata.",
    )
    canAcceptOwnership: bool | None = Field(
        None,
        description="Output only. Whether the current user is the pending owner of the file. Not populated for shared drive files.",
    )
    canReadLabels: bool | None = Field(
        None, description="Output only. Whether the current user can read the labels on the file."
    )
    canModifyLabels: bool | None = Field(
        None, description="Output only. Whether the current user can modify the labels on the file."
    )
    canModifyEditorContentRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can add or modify content restrictions on the file which are editor restricted.",
    )
    canModifyOwnerContentRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can add or modify content restrictions which are owner restricted.",
    )
    canRemoveContentRestriction: bool | None = Field(
        None,
        description="Output only. Whether there's a content restriction on the file that can be removed by the current user.",
    )
    canDisableInheritedPermissions: bool | None = Field(
        None, description="Whether a user can disable inherited permissions."
    )
    canEnableInheritedPermissions: bool | None = Field(
        None, description="Whether a user can re-enable inherited permissions."
    )
    canChangeItemDownloadRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the owner or organizer-applied download restrictions of the file.",
    )


class Location(BaseModel):
    latitude: float | None = Field(None, description="Output only. The latitude stored in the image.")
    longitude: float | None = Field(None, description="Output only. The longitude stored in the image.")
    altitude: float | None = Field(None, description="Output only. The altitude stored in the image.")


class ImageMediaMetadata(BaseModel):
    flashUsed: bool | None = Field(None, description="Output only. Whether a flash was used to create the photo.")
    meteringMode: str | None = Field(None, description="Output only. The metering mode used to create the photo.")
    sensor: str | None = Field(None, description="Output only. The type of sensor used to create the photo.")
    exposureMode: str | None = Field(None, description="Output only. The exposure mode used to create the photo.")
    colorSpace: str | None = Field(None, description="Output only. The color space of the photo.")
    whiteBalance: str | None = Field(None, description="Output only. The white balance mode used to create the photo.")
    width: int | None = Field(None, description="Output only. The width of the image in pixels.")
    height: int | None = Field(None, description="Output only. The height of the image in pixels.")
    location: Location | None = Field(
        None, description="Output only. Geographic location information stored in the image."
    )
    rotation: int | None = Field(
        None,
        description="Output only. The number of clockwise 90 degree rotations applied from the image's original orientation.",
    )
    time: str | None = Field(None, description="Output only. The date and time the photo was taken (EXIF DateTime).")
    cameraMake: str | None = Field(None, description="Output only. The make of the camera used to create the photo.")
    cameraModel: str | None = Field(None, description="Output only. The model of the camera used to create the photo.")
    exposureTime: float | None = Field(None, description="Output only. The length of the exposure, in seconds.")
    aperture: float | None = Field(None, description="Output only. The aperture used to create the photo (f-number).")
    focalLength: float | None = Field(
        None, description="Output only. The focal length used to create the photo, in millimeters."
    )
    isoSpeed: int | None = Field(None, description="Output only. The ISO speed used to create the photo.")
    exposureBias: float | None = Field(None, description="Output only. The exposure bias of the photo (APEX value).")
    maxApertureValue: float | None = Field(
        None,
        description="Output only. The smallest f-number of the lens at the focal length used to create the photo (APEX value).",
    )
    subjectDistance: int | None = Field(
        None, description="Output only. The distance to the subject of the photo, in meters."
    )
    lens: str | None = Field(None, description="Output only. The lens used to create the photo.")


class VideoMediaMetadata(BaseModel):
    width: int | None = Field(None, description="Output only. The width of the video in pixels.")
    height: int | None = Field(None, description="Output only. The height of the video in pixels.")
    durationMillis: int | None = Field(None, description="Output only. The duration of the video in milliseconds.")


class ShortcutDetails(BaseModel):
    targetId: str | None = Field(
        None, description="The ID of the file that this shortcut points to. Can only be set on `files.create` requests."
    )
    targetMimeType: str | None = Field(
        None,
        description="Output only. The MIME type of the file that this shortcut points to. The value of this field is a snapshot of the target's MIME type, captured when the shortcut is created.",
    )
    targetResourceKey: str | None = Field(None, description="Output only. The `resourceKey` for the target file.")


class LinkShareMetadata(BaseModel):
    securityUpdateEligible: bool | None = Field(
        None, description="Output only. Whether the file is eligible for security update."
    )
    securityUpdateEnabled: bool | None = Field(
        None, description="Output only. Whether the security update is enabled for this file."
    )


class PermissionDetail(BaseModel):
    permissionType: str | None = Field(
        None,
        description="Output only. The permission type for this user. While new values may be added in future, the following are currently possible: * `file` * `member`",
    )
    inheritedFrom: str | None = Field(
        None,
        description="Output only. The ID of the item from which this permission is inherited. This is only populated for items in shared drives.",
    )
    role: str | None = Field(
        None,
        description="Output only. The primary role for this user. While new values may be added in the future, the following are currently possible: * `owner` * `organizer` * `fileOrganizer` * `writer` * `commenter` * `reader`",
    )
    inherited: bool | None = Field(
        None,
        description="Output only. Whether this permission is inherited. This field is always populated. This is an output-only field.",
    )


class TeamDrivePermissionDetail(BaseModel):
    teamDrivePermissionType: str | None = Field(
        None, description="Deprecated: Output only. Use `permissionDetails/permissionType` instead."
    )
    inheritedFrom: str | None = Field(
        None, description="Deprecated: Output only. Use `permissionDetails/inheritedFrom` instead."
    )
    role: str | None = Field(None, description="Deprecated: Output only. Use `permissionDetails/role` instead.")
    inherited: bool | None = Field(
        None, description="Deprecated: Output only. Use `permissionDetails/inherited` instead."
    )


class Permission(BaseModel):
    id: str | None = Field(
        None,
        description="Output only. The ID of this permission. This is a unique identifier for the grantee, and is published in User resources as `permissionId`. IDs should be treated as opaque values.",
    )
    displayName: str | None = Field(
        None,
        description='Output only. The "pretty" name of the value of the permission. The following is a list of examples for each type of permission: * `user` - User\'s full name, as defined for their Google account, such as "Joe Smith." * `group` - Name of the Google Group, such as "The Company Administrators." * `domain` - String domain name, such as "thecompany.com." * `anyone` - No `displayName` is present.',
    )
    type: str | None = Field(
        None,
        description="The type of the grantee. Valid values are: * `user` * `group` * `domain` * `anyone` When creating a permission, if `type` is `user` or `group`, you must provide an `emailAddress` for the user or group. When `type` is `domain`, you must provide a `domain`. There isn't extra information required for an `anyone` type.",
    )
    kind: str | None = Field(
        "drive#permission",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#permission"`.',
    )
    permissionDetails: list[PermissionDetail] | None = Field(
        None,
        description="Output only. Details of whether the permissions on this item are inherited or directly on this item.",
    )
    photoLink: str | None = Field(None, description="Output only. A link to the user's profile photo, if available.")
    emailAddress: str | None = Field(
        None, description="The email address of the user or group to which this permission refers."
    )
    role: str | None = Field(
        None,
        description="The role granted by this permission. While new values may be supported in the future, the following are currently allowed: * `owner` * `organizer` * `fileOrganizer` * `writer` * `commenter` * `reader`",
    )
    allowFileDiscovery: bool | None = Field(
        None,
        description="Whether the permission allows the file to be discovered through search. This is only applicable for permissions of type `domain` or `anyone`.",
    )
    domain: str | None = Field(None, description="The domain to which this permission refers.")
    expirationTime: AwareDatetime | None = Field(
        None,
        description="The time at which this permission will expire (RFC 3339 date-time). Expiration times have the following restrictions: - They can only be set on user and group permissions - The time must be in the future - The time cannot be more than a year in the future",
    )
    teamDrivePermissionDetails: list[TeamDrivePermissionDetail] | None = Field(
        None, description="Output only. Deprecated: Output only. Use `permissionDetails` instead."
    )
    deleted: bool | None = Field(
        None,
        description="Output only. Whether the account associated with this permission has been deleted. This field only pertains to user and group permissions.",
    )
    view: str | None = Field(
        None,
        description="Indicates the view for this permission. Only populated for permissions that belong to a view. published and metadata are the only supported values. - published: The permission's role is published_reader. - metadata: The item is only visible to the metadata view because the item has limited access and the scope has at least read access to the parent. Note: The metadata view is currently only supported on folders. ",
    )
    pendingOwner: bool | None = Field(
        None,
        description="Whether the account associated with this permission is a pending owner. Only populated for `user` type permissions for files that are not in a shared drive.",
    )
    inheritedPermissionsDisabled: bool | None = Field(
        None,
        description="When true, only organizers, owners, and users with permissions added directly on the item can access it.",
    )


class ContentRestriction(BaseModel):
    readOnly: bool | None = Field(
        None,
        description="Whether the content of the file is read-only. If a file is read-only, a new revision of the file may not be added, comments may not be added or modified, and the title of the file may not be modified.",
    )
    reason: str | None = Field(
        None,
        description="Reason for why the content of the file is restricted. This is only mutable on requests that also set `readOnly=true`.",
    )
    type: str | None = Field(
        None,
        description="Output only. The type of the content restriction. Currently the only possible value is `globalContentRestriction`.",
    )
    restrictingUser: User | None = Field(
        None, description="Output only. The user who set the content restriction. Only populated if `readOnly=true`."
    )
    restrictionTime: AwareDatetime | None = Field(
        None,
        description="The time at which the content restriction was set (formatted RFC 3339 timestamp). Only populated if readOnly is true.",
    )
    ownerRestricted: bool | None = Field(
        None,
        description="Whether the content restriction can only be modified or removed by a user who owns the file. For files in shared drives, any user with `organizer` capabilities can modify or remove this content restriction.",
    )
    systemRestricted: bool | None = Field(
        None,
        description="Output only. Whether the content restriction was applied by the system, for example due to an esignature. Users cannot modify or remove system restricted content restrictions.",
    )


class LabelField(BaseModel):
    kind: str | None = Field(None, description="This is always drive#labelField.")
    id: str | None = Field(None, description="The identifier of this label field.")
    valueType: str | None = Field(
        None,
        description="The field type. While new values may be supported in the future, the following are currently allowed: * `dateString` * `integer` * `selection` * `text` * `user`",
    )
    dateString: list[date] | None = Field(
        None, description="Only present if valueType is dateString. RFC 3339 formatted date: YYYY-MM-DD."
    )
    integer: list[int] | None = Field(None, description="Only present if `valueType` is `integer`.")
    selection: list[str] | None = Field(None, description="Only present if `valueType` is `selection`")
    text: list[str] | None = Field(None, description="Only present if `valueType` is `text`.")
    user: list[User] | None = Field(None, description="Only present if `valueType` is `user`.")


class DownloadRestriction(BaseModel):
    restrictedForReaders: bool | None = Field(None, description="Whether download and copy is restricted for readers.")
    restrictedForWriters: bool | None = Field(
        None,
        description="Whether download and copy is restricted for writers. If `true`, download is also restricted for readers.",
    )


class Capabilities1(BaseModel):
    canAddChildren: bool | None = Field(
        None, description="Whether the current user can add children to folders in this Team Drive."
    )
    canComment: bool | None = Field(
        None, description="Whether the current user can comment on files in this Team Drive."
    )
    canCopy: bool | None = Field(None, description="Whether the current user can copy files in this Team Drive.")
    canDeleteTeamDrive: bool | None = Field(
        None,
        description="Whether the current user can delete this Team Drive. Attempting to delete the Team Drive may still fail if there are untrashed items inside the Team Drive.",
    )
    canDownload: bool | None = Field(
        None, description="Whether the current user can download files in this Team Drive."
    )
    canEdit: bool | None = Field(None, description="Whether the current user can edit files in this Team Drive")
    canListChildren: bool | None = Field(
        None, description="Whether the current user can list the children of folders in this Team Drive."
    )
    canManageMembers: bool | None = Field(
        None,
        description="Whether the current user can add members to this Team Drive or remove them or change their role.",
    )
    canReadRevisions: bool | None = Field(
        None, description="Whether the current user can read the revisions resource of files in this Team Drive."
    )
    canRemoveChildren: bool | None = Field(
        None, description="Deprecated: Use `canDeleteChildren` or `canTrashChildren` instead."
    )
    canRename: bool | None = Field(
        None, description="Whether the current user can rename files or folders in this Team Drive."
    )
    canRenameTeamDrive: bool | None = Field(None, description="Whether the current user can rename this Team Drive.")
    canChangeTeamDriveBackground: bool | None = Field(
        None, description="Whether the current user can change the background of this Team Drive."
    )
    canShare: bool | None = Field(
        None, description="Whether the current user can share files or folders in this Team Drive."
    )
    canChangeCopyRequiresWriterPermissionRestriction: bool | None = Field(
        None,
        description="Whether the current user can change the `copyRequiresWriterPermission` restriction of this Team Drive.",
    )
    canChangeDomainUsersOnlyRestriction: bool | None = Field(
        None, description="Whether the current user can change the `domainUsersOnly` restriction of this Team Drive."
    )
    canChangeSharingFoldersRequiresOrganizerPermissionRestriction: bool | None = Field(
        None,
        description="Whether the current user can change the `sharingFoldersRequiresOrganizerPermission` restriction of this Team Drive.",
    )
    canChangeTeamMembersOnlyRestriction: bool | None = Field(
        None, description="Whether the current user can change the `teamMembersOnly` restriction of this Team Drive."
    )
    canDeleteChildren: bool | None = Field(
        None, description="Whether the current user can delete children from folders in this Team Drive."
    )
    canTrashChildren: bool | None = Field(
        None, description="Whether the current user can trash children from folders in this Team Drive."
    )
    canResetTeamDriveRestrictions: bool | None = Field(
        None, description="Whether the current user can reset the Team Drive restrictions to defaults."
    )
    canChangeDownloadRestriction: bool | None = Field(
        None,
        description="Whether the current user can change organizer-applied download restrictions of this shared drive.",
    )


class BackgroundImageFile(BaseModel):
    id: str | None = Field(None, description="The ID of an image file in Drive to use for the background image.")
    xCoordinate: float | None = Field(
        None,
        description="The X coordinate of the upper left corner of the cropping area in the background image. This is a value in the closed range of 0 to 1. This value represents the horizontal distance from the left side of the entire image to the left side of the cropping area divided by the width of the entire image.",
    )
    yCoordinate: float | None = Field(
        None,
        description="The Y coordinate of the upper left corner of the cropping area in the background image. This is a value in the closed range of 0 to 1. This value represents the vertical distance from the top side of the entire image to the top side of the cropping area divided by the height of the entire image.",
    )
    width: float | None = Field(
        None,
        description="The width of the cropped image in the closed range of 0 to 1. This value represents the width of the cropped image divided by the width of the entire image. The height is computed by applying a width to height aspect ratio of 80 to 9. The resulting image must be at least 1280 pixels wide and 144 pixels high.",
    )


class Restrictions(BaseModel):
    copyRequiresWriterPermission: bool | None = Field(
        None,
        description="Whether the options to copy, print, or download files inside this Team Drive, should be disabled for readers and commenters. When this restriction is set to `true`, it will override the similarly named field to `true` for any file inside this Team Drive.",
    )
    domainUsersOnly: bool | None = Field(
        None,
        description="Whether access to this Team Drive and items inside this Team Drive is restricted to users of the domain to which this Team Drive belongs. This restriction may be overridden by other sharing policies controlled outside of this Team Drive.",
    )
    teamMembersOnly: bool | None = Field(
        None, description="Whether access to items inside this Team Drive is restricted to members of this Team Drive."
    )
    adminManagedRestrictions: bool | None = Field(
        None, description="Whether administrative privileges on this Team Drive are required to modify restrictions."
    )
    sharingFoldersRequiresOrganizerPermission: bool | None = Field(
        None,
        description="If true, only users with the organizer role can share folders. If false, users with either the organizer role or the file organizer role can share folders.",
    )
    downloadRestriction: DownloadRestriction | None = Field(
        None, description="Download restrictions applied by shared drive managers."
    )


class TeamDrive(BaseModel):
    id: str | None = Field(
        None, description="The ID of this Team Drive which is also the ID of the top level folder of this Team Drive."
    )
    name: str | None = Field(None, description="The name of this Team Drive.")
    colorRgb: str | None = Field(
        None,
        description="The color of this Team Drive as an RGB hex string. It can only be set on a `drive.teamdrives.update` request that does not set `themeId`.",
    )
    kind: str | None = Field(
        "drive#teamDrive",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#teamDrive"`.',
    )
    backgroundImageLink: str | None = Field(
        None, description="A short-lived link to this Team Drive's background image."
    )
    capabilities: Capabilities1 | None = Field(
        None, description="Capabilities the current user has on this Team Drive."
    )
    themeId: str | None = Field(
        None,
        description="The ID of the theme from which the background image and color will be set. The set of possible `teamDriveThemes` can be retrieved from a `drive.about.get` response. When not specified on a `drive.teamdrives.create` request, a random theme is chosen from which the background image and color are set. This is a write-only field; it can only be set on requests that don't set `colorRgb` or `backgroundImageFile`.",
    )
    backgroundImageFile: BackgroundImageFile | None = Field(
        None,
        description="An image file and cropping parameters from which a background image for this Team Drive is set. This is a write only field; it can only be set on `drive.teamdrives.update` requests that don't set `themeId`. When specified, all fields of the `backgroundImageFile` must be set.",
    )
    createdTime: AwareDatetime | None = Field(
        None, description="The time at which the Team Drive was created (RFC 3339 date-time)."
    )
    restrictions: Restrictions | None = Field(
        None, description="A set of restrictions that apply to this Team Drive or items inside this Team Drive."
    )
    orgUnitId: str | None = Field(
        None,
        description="The organizational unit of this shared drive. This field is only populated on `drives.list` responses when the `useDomainAdminAccess` parameter is set to `true`.",
    )


class Capabilities2(BaseModel):
    canAddChildren: bool | None = Field(
        None, description="Output only. Whether the current user can add children to folders in this shared drive."
    )
    canComment: bool | None = Field(
        None, description="Output only. Whether the current user can comment on files in this shared drive."
    )
    canCopy: bool | None = Field(
        None, description="Output only. Whether the current user can copy files in this shared drive."
    )
    canDeleteDrive: bool | None = Field(
        None,
        description="Output only. Whether the current user can delete this shared drive. Attempting to delete the shared drive may still fail if there are untrashed items inside the shared drive.",
    )
    canDownload: bool | None = Field(
        None, description="Output only. Whether the current user can download files in this shared drive."
    )
    canEdit: bool | None = Field(
        None, description="Output only. Whether the current user can edit files in this shared drive"
    )
    canListChildren: bool | None = Field(
        None, description="Output only. Whether the current user can list the children of folders in this shared drive."
    )
    canManageMembers: bool | None = Field(
        None,
        description="Output only. Whether the current user can add members to this shared drive or remove them or change their role.",
    )
    canReadRevisions: bool | None = Field(
        None,
        description="Output only. Whether the current user can read the revisions resource of files in this shared drive.",
    )
    canRename: bool | None = Field(
        None, description="Output only. Whether the current user can rename files or folders in this shared drive."
    )
    canRenameDrive: bool | None = Field(
        None, description="Output only. Whether the current user can rename this shared drive."
    )
    canChangeDriveBackground: bool | None = Field(
        None, description="Output only. Whether the current user can change the background of this shared drive."
    )
    canShare: bool | None = Field(
        None, description="Output only. Whether the current user can share files or folders in this shared drive."
    )
    canChangeCopyRequiresWriterPermissionRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `copyRequiresWriterPermission` restriction of this shared drive.",
    )
    canChangeDomainUsersOnlyRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `domainUsersOnly` restriction of this shared drive.",
    )
    canChangeDriveMembersOnlyRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `driveMembersOnly` restriction of this shared drive.",
    )
    canChangeSharingFoldersRequiresOrganizerPermissionRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change the `sharingFoldersRequiresOrganizerPermission` restriction of this shared drive.",
    )
    canResetDriveRestrictions: bool | None = Field(
        None, description="Output only. Whether the current user can reset the shared drive restrictions to defaults."
    )
    canDeleteChildren: bool | None = Field(
        None, description="Output only. Whether the current user can delete children from folders in this shared drive."
    )
    canTrashChildren: bool | None = Field(
        None, description="Output only. Whether the current user can trash children from folders in this shared drive."
    )
    canChangeDownloadRestriction: bool | None = Field(
        None,
        description="Output only. Whether the current user can change organizer-applied download restrictions of this shared drive.",
    )


class BackgroundImageFile1(BaseModel):
    id: str | None = Field(None, description="The ID of an image file in Google Drive to use for the background image.")
    xCoordinate: float | None = Field(
        None,
        description="The X coordinate of the upper left corner of the cropping area in the background image. This is a value in the closed range of 0 to 1. This value represents the horizontal distance from the left side of the entire image to the left side of the cropping area divided by the width of the entire image.",
    )
    yCoordinate: float | None = Field(
        None,
        description="The Y coordinate of the upper left corner of the cropping area in the background image. This is a value in the closed range of 0 to 1. This value represents the vertical distance from the top side of the entire image to the top side of the cropping area divided by the height of the entire image.",
    )
    width: float | None = Field(
        None,
        description="The width of the cropped image in the closed range of 0 to 1. This value represents the width of the cropped image divided by the width of the entire image. The height is computed by applying a width to height aspect ratio of 80 to 9. The resulting image must be at least 1280 pixels wide and 144 pixels high.",
    )


class Restrictions1(BaseModel):
    copyRequiresWriterPermission: bool | None = Field(
        None,
        description="Whether the options to copy, print, or download files inside this shared drive, should be disabled for readers and commenters. When this restriction is set to `true`, it will override the similarly named field to `true` for any file inside this shared drive.",
    )
    domainUsersOnly: bool | None = Field(
        None,
        description="Whether access to this shared drive and items inside this shared drive is restricted to users of the domain to which this shared drive belongs. This restriction may be overridden by other sharing policies controlled outside of this shared drive.",
    )
    driveMembersOnly: bool | None = Field(
        None, description="Whether access to items inside this shared drive is restricted to its members."
    )
    adminManagedRestrictions: bool | None = Field(
        None, description="Whether administrative privileges on this shared drive are required to modify restrictions."
    )
    sharingFoldersRequiresOrganizerPermission: bool | None = Field(
        None,
        description="If true, only users with the organizer role can share folders. If false, users with either the organizer role or the file organizer role can share folders.",
    )
    downloadRestriction: DownloadRestriction | None = Field(
        None, description="Download restrictions applied by shared drive managers."
    )


class Drive(BaseModel):
    id: str | None = Field(
        None,
        description="Output only. The ID of this shared drive which is also the ID of the top level folder of this shared drive.",
    )
    name: str | None = Field(None, description="The name of this shared drive.")
    colorRgb: str | None = Field(
        None,
        description="The color of this shared drive as an RGB hex string. It can only be set on a `drive.drives.update` request that does not set `themeId`.",
    )
    kind: str | None = Field(
        "drive#drive",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#drive"`.',
    )
    backgroundImageLink: str | None = Field(
        None, description="Output only. A short-lived link to this shared drive's background image."
    )
    capabilities: Capabilities2 | None = Field(
        None, description="Output only. Capabilities the current user has on this shared drive."
    )
    themeId: str | None = Field(
        None,
        description="The ID of the theme from which the background image and color will be set. The set of possible `driveThemes` can be retrieved from a `drive.about.get` response. When not specified on a `drive.drives.create` request, a random theme is chosen from which the background image and color are set. This is a write-only field; it can only be set on requests that don't set `colorRgb` or `backgroundImageFile`.",
    )
    backgroundImageFile: BackgroundImageFile1 | None = Field(
        None,
        description="An image file and cropping parameters from which a background image for this shared drive is set. This is a write only field; it can only be set on `drive.drives.update` requests that don't set `themeId`. When specified, all fields of the `backgroundImageFile` must be set.",
    )
    createdTime: AwareDatetime | None = Field(
        None, description="The time at which the shared drive was created (RFC 3339 date-time)."
    )
    hidden: bool | None = Field(None, description="Whether the shared drive is hidden from default view.")
    restrictions: Restrictions1 | None = Field(
        None,
        description="A set of restrictions that apply to this shared drive or items inside this shared drive. Note that restrictions can't be set when creating a shared drive. To add a restriction, first create a shared drive and then use `drives.update` to add restrictions.",
    )
    orgUnitId: str | None = Field(
        None,
        description="Output only. The organizational unit of this shared drive. This field is only populated on `drives.list` responses when the `useDomainAdminAccess` parameter is set to `true`.",
    )


class Channel(BaseModel):
    payload: bool | None = Field(None, description="A Boolean value to indicate whether payload is wanted. Optional.")
    id: str | None = Field(None, description="A UUID or similar unique string that identifies this channel.")
    resourceId: str | None = Field(
        None,
        description="An opaque ID that identifies the resource being watched on this channel. Stable across different API versions.",
    )
    resourceUri: str | None = Field(None, description="A version-specific identifier for the watched resource.")
    token: str | None = Field(
        None,
        description="An arbitrary string delivered to the target address with each notification delivered over this channel. Optional.",
    )
    expiration: int | None = Field(
        None,
        description="Date and time of notification channel expiration, expressed as a Unix timestamp, in milliseconds. Optional.",
    )
    type: str | None = Field(
        None,
        description='The type of delivery mechanism used for this channel. Valid values are "web_hook" or "webhook".',
    )
    address: str | None = Field(None, description="The address where notifications are delivered for this channel.")
    params: dict[str, str] | None = Field(
        None, description="Additional parameters controlling delivery channel behavior. Optional."
    )
    kind: str | None = Field(
        "api#channel",
        description="Identifies this as a notification channel used to watch for changes to a resource, which is `api#channel`.",
    )


class QuotedFileContent(BaseModel):
    mimeType: str | None = Field(None, description="The MIME type of the quoted content.")
    value: str | None = Field(
        None, description="The quoted content itself. This is interpreted as plain text if set through the API."
    )


class Reply(BaseModel):
    id: str | None = Field(None, description="Output only. The ID of the reply.")
    kind: str | None = Field(
        "drive#reply",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#reply"`.',
    )
    createdTime: AwareDatetime | None = Field(
        None, description="The time at which the reply was created (RFC 3339 date-time)."
    )
    modifiedTime: AwareDatetime | None = Field(
        None, description="The last time the reply was modified (RFC 3339 date-time)."
    )
    action: str | None = Field(
        None,
        description="The action the reply performed to the parent comment. Valid values are: * `resolve` * `reopen`",
    )
    author: User | None = Field(
        None,
        description="Output only. The author of the reply. The author's email address and permission ID will not be populated.",
    )
    deleted: bool | None = Field(
        None, description="Output only. Whether the reply has been deleted. A deleted reply has no content."
    )
    htmlContent: str | None = Field(None, description="Output only. The content of the reply with HTML formatting.")
    content: str | None = Field(
        None,
        description="The plain text content of the reply. This field is used for setting the content, while `htmlContent` should be displayed. This is required on creates if no `action` is specified.",
    )


class DriveList(BaseModel):
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of shared drives. This will be absent if the end of the list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(
        "drive#driveList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#driveList"`.',
    )
    drives: list[Drive] | None = Field(
        None,
        description="The list of shared drives. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )


class GeneratedIds(BaseModel):
    ids: list[str] | None = Field(None, description="The IDs generated for the requesting user in the specified space.")
    space: str | None = Field(None, description="The type of file that can be created with these IDs.")
    kind: str | None = Field(
        "drive#generatedIds",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#generatedIds"`.',
    )


class LabelFieldModification(BaseModel):
    fieldId: str | None = Field(None, description="The ID of the field to be modified.")
    kind: str | None = Field(None, description='This is always `"drive#labelFieldModification"`.')
    setDateValues: list[date] | None = Field(
        None,
        description="Replaces the value of a dateString Field with these new values. The string must be in the RFC 3339 full-date format: YYYY-MM-DD.",
    )
    setTextValues: list[str] | None = Field(None, description="Sets the value of a `text` field.")
    setSelectionValues: list[str] | None = Field(
        None, description="Replaces a `selection` field with these new values."
    )
    setIntegerValues: list[int] | None = Field(
        None, description="Replaces the value of an `integer` field with these new values."
    )
    setUserValues: list[str] | None = Field(
        None, description="Replaces a `user` field with these new values. The values must be a valid email addresses."
    )
    unsetValues: bool | None = Field(None, description="Unsets the values for this field.")


class PermissionList(BaseModel):
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of permissions. This field will be absent if the end of the permissions list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(
        "drive#permissionList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#permissionList"`.',
    )
    permissions: list[Permission] | None = Field(
        None,
        description="The list of permissions. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )


class ReplyList(BaseModel):
    kind: str | None = Field(
        "drive#replyList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#replyList"`.',
    )
    replies: list[Reply] | None = Field(
        None,
        description="The list of replies. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of replies. This will be absent if the end of the replies list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )


class Revision(BaseModel):
    id: str | None = Field(None, description="Output only. The ID of the revision.")
    mimeType: str | None = Field(None, description="Output only. The MIME type of the revision.")
    kind: str | None = Field(
        "drive#revision",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#revision"`.',
    )
    published: bool | None = Field(
        None, description="Whether this revision is published. This is only applicable to Docs Editors files."
    )
    exportLinks: dict[str, str] | None = Field(
        None, description="Output only. Links for exporting Docs Editors files to specific formats."
    )
    keepForever: bool | None = Field(
        None,
        description="Whether to keep this revision forever, even if it is no longer the head revision. If not set, the revision will be automatically purged 30 days after newer content is uploaded. This can be set on a maximum of 200 revisions for a file. This field is only applicable to files with binary content in Drive.",
    )
    md5Checksum: str | None = Field(
        None,
        description="Output only. The MD5 checksum of the revision's content. This is only applicable to files with binary content in Drive.",
    )
    modifiedTime: AwareDatetime | None = Field(
        None, description="The last time the revision was modified (RFC 3339 date-time)."
    )
    publishAuto: bool | None = Field(
        None,
        description="Whether subsequent revisions will be automatically republished. This is only applicable to Docs Editors files.",
    )
    publishedOutsideDomain: bool | None = Field(
        None,
        description="Whether this revision is published outside the domain. This is only applicable to Docs Editors files.",
    )
    publishedLink: str | None = Field(
        None,
        description="Output only. A link to the published revision. This is only populated for Docs Editors files.",
    )
    size: int | None = Field(
        None,
        description="Output only. The size of the revision's content in bytes. This is only applicable to files with binary content in Drive.",
    )
    originalFilename: str | None = Field(
        None,
        description="Output only. The original filename used to create this revision. This is only applicable to files with binary content in Drive.",
    )
    lastModifyingUser: User | None = Field(
        None,
        description="Output only. The last user to modify this revision. This field is only populated when the last modification was performed by a signed-in user.",
    )


class RevisionList(BaseModel):
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of revisions. This will be absent if the end of the revisions list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(
        "drive#revisionList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#revisionList"`.',
    )
    revisions: list[Revision] | None = Field(
        None,
        description="The list of revisions. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )


class TeamDriveList(BaseModel):
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of Team Drives. This will be absent if the end of the Team Drives list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(
        "drive#teamDriveList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#teamDriveList"`.',
    )
    teamDrives: list[TeamDrive] | None = Field(
        None,
        description="The list of Team Drives. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )


class AccessProposalRoleAndView(BaseModel):
    role: str | None = Field(
        None,
        description="The role that was proposed by the requester. The supported values are: * `writer` * `commenter` * `reader`",
    )
    view: str | None = Field(
        None,
        description="Indicates the view for this access proposal. Only populated for proposals that belong to a view. Only `published` is supported.",
    )


class Action(Enum):
    ACTION_UNSPECIFIED = "ACTION_UNSPECIFIED"
    ACCEPT = "ACCEPT"
    DENY = "DENY"


class ResolveAccessProposalRequest(BaseModel):
    class Config:
        use_enum_values = True

    role: list[str] | None = Field(
        None,
        description="Optional. The roles that the approver has allowed, if any. For more information, see [Roles and permissions](https://developers.google.com/workspace/drive/api/guides/ref-roles). Note: This field is required for the `ACCEPT` action.",
    )
    view: str | None = Field(
        None,
        description="Optional. Indicates the view for this access proposal. This should only be set when the proposal belongs to a view. Only `published` is supported.",
    )
    action: Action | None = Field(None, description="Required. The action to take on the access proposal.")
    sendNotification: bool | None = Field(
        None,
        description="Optional. Whether to send an email to the requester when the access proposal is denied or accepted.",
    )


class Operation(BaseModel):
    name: str | None = Field(
        None,
        description="The server-assigned name, which is only unique within the same service that originally returns it. If you use the default HTTP mapping, the `name` should be a resource name ending with `operations/{unique_id}`.",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Service-specific metadata associated with the operation. It typically contains progress information and common metadata such as create time. Some services might not provide such metadata. Any method that returns a long-running operation should document the metadata type, if any.",
    )
    done: bool | None = Field(
        None,
        description="If the value is `false`, it means the operation is still in progress. If `true`, the operation is completed, and either `error` or `response` is available.",
    )
    error: Status | None = Field(
        None, description="The error result of the operation in case of failure or cancellation."
    )
    response: dict[str, Any] | None = Field(
        None,
        description="The normal, successful response of the operation. If the original method returns no data on success, such as `Delete`, the response is `google.protobuf.Empty`. If the original method is standard `Get`/`Create`/`Update`, the response should be the resource. For other methods, the response should have the type `XxxResponse`, where `Xxx` is the original method name. For example, if the original method name is `TakeSnapshot()`, the inferred response type is `TakeSnapshotResponse`.",
    )


class About(BaseModel):
    kind: str | None = Field(
        "drive#about", description='Identifies what kind of resource this is. Value: the fixed string `"drive#about"`.'
    )
    storageQuota: StorageQuota | None = Field(
        None,
        description="The user's storage quota limits and usage. For users that are part of an organization with pooled storage, information about the limit and usage across all services is for the organization, rather than the individual user. All fields are measured in bytes.",
    )
    driveThemes: list[DriveTheme] | None = Field(
        None, description="A list of themes that are supported for shared drives."
    )
    canCreateDrives: bool | None = Field(None, description="Whether the user can create shared drives.")
    importFormats: dict[str, list[str]] | None = Field(
        None, description="A map of source MIME type to possible targets for all supported imports."
    )
    exportFormats: dict[str, list[str]] | None = Field(
        None, description="A map of source MIME type to possible targets for all supported exports."
    )
    appInstalled: bool | None = Field(None, description="Whether the user has installed the requesting app.")
    user: User | None = Field(None, description="The authenticated user.")
    folderColorPalette: list[str] | None = Field(
        None, description="The currently supported folder colors as RGB hex strings."
    )
    maxImportSizes: dict[str, int] | None = Field(
        None, description="A map of maximum import sizes by MIME type, in bytes."
    )
    maxUploadSize: int | None = Field(None, description="The maximum upload size in bytes.")
    teamDriveThemes: list[TeamDriveTheme] | None = Field(None, description="Deprecated: Use `driveThemes` instead.")
    canCreateTeamDrives: bool | None = Field(None, description="Deprecated: Use `canCreateDrives` instead.")


class App(BaseModel):
    name: str | None = Field(None, description="The name of the app.")
    objectType: str | None = Field(
        None,
        description="The type of object this app creates such as a Chart. If empty, the app name should be used instead.",
    )
    supportsCreate: bool | None = Field(None, description="Whether this app supports creating objects.")
    productUrl: str | None = Field(None, description="A link to the product listing for this app.")
    primaryMimeTypes: list[str] | None = Field(None, description="The list of primary MIME types.")
    secondaryMimeTypes: list[str] | None = Field(None, description="The list of secondary MIME types.")
    primaryFileExtensions: list[str] | None = Field(None, description="The list of primary file extensions.")
    secondaryFileExtensions: list[str] | None = Field(None, description="The list of secondary file extensions.")
    id: str | None = Field(None, description="The ID of the app.")
    supportsImport: bool | None = Field(None, description="Whether this app supports importing from Google Docs.")
    installed: bool | None = Field(None, description="Whether the app is installed.")
    authorized: bool | None = Field(
        None, description="Whether the app is authorized to access data on the user's Drive."
    )
    icons: list[AppIcons] | None = Field(None, description="The various icons for the app.")
    useByDefault: bool | None = Field(
        None, description="Whether the app is selected as the default handler for the types it supports."
    )
    kind: str | None = Field(
        "drive#app",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string "drive#app".',
    )
    shortDescription: str | None = Field(None, description="A short description of the app.")
    longDescription: str | None = Field(None, description="A long description of the app.")
    supportsMultiOpen: bool | None = Field(None, description="Whether this app supports opening more than one file.")
    productId: str | None = Field(None, description="The ID of the product listing for this app.")
    openUrlTemplate: str | None = Field(
        None,
        description="The template URL for opening files with this app. The template contains {ids} or {exportIds} to be replaced by the actual file IDs. For more information, see Open Files for the full documentation.",
    )
    createUrl: str | None = Field(None, description="The URL to create a file with this app.")
    createInFolderTemplate: str | None = Field(
        None,
        description="The template URL to create a file with this app in a given folder. The template contains the {folderId} to be replaced by the folder ID house the new file.",
    )
    supportsOfflineCreate: bool | None = Field(
        None, description="Whether this app supports creating files when offline."
    )
    hasDriveWideScope: bool | None = Field(
        None,
        description="Whether the app has Drive-wide scope. An app with Drive-wide scope can access all files in the user's Drive.",
    )


class AppList(BaseModel):
    defaultAppIds: list[str] | None = Field(
        None,
        description="The list of app IDs that the user has specified to use by default. The list is in reverse-priority order (lowest to highest).",
    )
    kind: str | None = Field(
        "drive#appList",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string "drive#appList".',
    )
    selfLink: str | None = Field(None, description="A link back to this list.")
    items: list[App] | None = Field(None, description="The list of apps.")


class Label(BaseModel):
    id: str | None = Field(None, description="The ID of the label.")
    revisionId: str | None = Field(None, description="The revision ID of the label.")
    kind: str | None = Field(None, description="This is always drive#label")
    fields: dict[str, LabelField] | None = Field(
        None, description="A map of the fields on the label, keyed by the field's ID."
    )


class DownloadRestrictionsMetadata(BaseModel):
    itemDownloadRestriction: DownloadRestriction | None = Field(
        None,
        description="The download restriction of the file applied directly by the owner or organizer. This doesn't take into account shared drive settings or DLP rules.",
    )
    effectiveDownloadRestrictionWithContext: DownloadRestriction | None = Field(
        None,
        description="Output only. The effective download restriction applied to this file. This considers all restriction settings and DLP rules.",
    )


class Comment(BaseModel):
    id: str | None = Field(None, description="Output only. The ID of the comment.")
    kind: str | None = Field(
        "drive#comment",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#comment"`.',
    )
    createdTime: AwareDatetime | None = Field(
        None, description="The time at which the comment was created (RFC 3339 date-time)."
    )
    modifiedTime: AwareDatetime | None = Field(
        None, description="The last time the comment or any of its replies was modified (RFC 3339 date-time)."
    )
    resolved: bool | None = Field(
        None, description="Output only. Whether the comment has been resolved by one of its replies."
    )
    anchor: str | None = Field(
        None,
        description="A region of the document represented as a JSON string. For details on defining anchor properties, refer to [Manage comments and replies](https://developers.google.com/workspace/drive/api/v3/manage-comments).",
    )
    replies: list[Reply] | None = Field(
        None, description="Output only. The full list of replies to the comment in chronological order."
    )
    author: User | None = Field(
        None,
        description="Output only. The author of the comment. The author's email address and permission ID will not be populated.",
    )
    deleted: bool | None = Field(
        None, description="Output only. Whether the comment has been deleted. A deleted comment has no content."
    )
    htmlContent: str | None = Field(None, description="Output only. The content of the comment with HTML formatting.")
    content: str | None = Field(
        None,
        description="The plain text content of the comment. This field is used for setting the content, while `htmlContent` should be displayed.",
    )
    quotedFileContent: QuotedFileContent | None = Field(
        None,
        description="The file content to which the comment refers, typically within the anchor region. For a text file, for example, this would be the text at the location of the comment.",
    )


class CommentList(BaseModel):
    kind: str | None = Field(
        "drive#commentList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#commentList"`.',
    )
    comments: list[Comment] | None = Field(
        None,
        description="The list of comments. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of comments. This will be absent if the end of the comments list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )


class LabelList(BaseModel):
    labels: list[Label] | None = Field(None, description="The list of labels.")
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of labels. This field will be absent if the end of the list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(None, description='This is always `"drive#labelList"`.')


class LabelModification(BaseModel):
    labelId: str | None = Field(None, description="The ID of the label to modify.")
    fieldModifications: list[LabelFieldModification] | None = Field(
        None, description="The list of modifications to this label's fields."
    )
    removeLabel: bool | None = Field(None, description="If true, the label will be removed from the file.")
    kind: str | None = Field(None, description='This is always `"drive#labelModification"`.')


class ModifyLabelsResponse(BaseModel):
    modifiedLabels: list[Label] | None = Field(
        None, description="The list of labels which were added or updated by the request."
    )
    kind: str | None = Field(None, description='This is always `"drive#modifyLabelsResponse"`.')


class AccessProposal(BaseModel):
    fileId: str | None = Field(None, description="The file ID that the proposal for access is on.")
    proposalId: str | None = Field(None, description="The ID of the access proposal.")
    requesterEmailAddress: str | None = Field(None, description="The email address of the requesting user.")
    recipientEmailAddress: str | None = Field(
        None, description="The email address of the user that will receive permissions, if accepted."
    )
    rolesAndViews: list[AccessProposalRoleAndView] | None = Field(
        None,
        description="A wrapper for the role and view of an access proposal. For more information, see [Roles and permissions](https://developers.google.com/workspace/drive/api/guides/ref-roles).",
    )
    requestMessage: str | None = Field(None, description="The message that the requester added to the proposal.")
    createTime: str | None = Field(None, description="The creation time.")


class ListAccessProposalsResponse(BaseModel):
    accessProposals: list[AccessProposal] | None = Field(
        None, description="The list of access proposals. This field is only populated in Drive API v3."
    )
    nextPageToken: str | None = Field(
        None,
        description="The continuation token for the next page of results. This will be absent if the end of the results list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results.",
    )


class LabelInfo(BaseModel):
    labels: list[Label] | None = Field(
        None,
        description="Output only. The set of labels on the file as requested by the label IDs in the `includeLabels` parameter. By default, no labels are returned.",
    )


class File(BaseModel):
    kind: str | None = Field(
        "drive#file",
        description='Output only. Identifies what kind of resource this is. Value: the fixed string `"drive#file"`.',
    )
    driveId: str | None = Field(
        None,
        description="Output only. ID of the shared drive the file resides in. Only populated for items in shared drives.",
    )
    fileExtension: str | None = Field(
        None,
        description="Output only. The final component of `fullFileExtension`. This is only available for files with binary content in Google Drive.",
    )
    copyRequiresWriterPermission: bool | None = Field(
        None,
        description="Whether the options to copy, print, or download this file should be disabled for readers and commenters.",
    )
    md5Checksum: str | None = Field(
        None,
        description="Output only. The MD5 checksum for the content of the file. This is only applicable to files with binary content in Google Drive.",
    )
    contentHints: ContentHints | None = Field(
        None,
        description="Additional information about the content of the file. These fields are never populated in responses.",
    )
    writersCanShare: bool | None = Field(
        None,
        description="Whether users with only `writer` permission can modify the file's permissions. Not populated for items in shared drives.",
    )
    viewedByMe: bool | None = Field(None, description="Output only. Whether the file has been viewed by this user.")
    mimeType: str | None = Field(
        None,
        description="The MIME type of the file. Google Drive attempts to automatically detect an appropriate value from uploaded content, if no value is provided. The value cannot be changed unless a new revision is uploaded. If a file is created with a Google Doc MIME type, the uploaded content is imported, if possible. The supported import formats are published in the [`about`](/workspace/drive/api/reference/rest/v3/about) resource.",
    )
    exportLinks: dict[str, str] | None = Field(
        None, description="Output only. Links for exporting Docs Editors files to specific formats."
    )
    parents: list[str] | None = Field(
        None,
        description="The ID of the parent folder containing the file. A file can only have one parent folder; specifying multiple parents isn't supported. If not specified as part of a create request, the file is placed directly in the user's My Drive folder. If not specified as part of a copy request, the file inherits any discoverable parent of the source file. Update requests must use the `addParents` and `removeParents` parameters to modify the parents list.",
    )
    thumbnailLink: str | None = Field(
        None,
        description="Output only. A short-lived link to the file's thumbnail, if available. Typically lasts on the order of hours. Not intended for direct usage on web applications due to [Cross-Origin Resource Sharing (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) policies. Consider using a proxy server. Only populated when the requesting app can access the file's content. If the file isn't shared publicly, the URL returned in `files.thumbnailLink` must be fetched using a credentialed request.",
    )
    iconLink: str | None = Field(None, description="Output only. A static, unauthenticated link to the file's icon.")
    shared: bool | None = Field(
        None, description="Output only. Whether the file has been shared. Not populated for items in shared drives."
    )
    lastModifyingUser: User | None = Field(
        None,
        description="Output only. The last user to modify the file. This field is only populated when the last modification was performed by a signed-in user.",
    )
    owners: list[User] | None = Field(
        None,
        description="Output only. The owner of this file. Only certain legacy files may have more than one owner. This field isn't populated for items in shared drives.",
    )
    headRevisionId: str | None = Field(
        None,
        description="Output only. The ID of the file's head revision. This is currently only available for files with binary content in Google Drive.",
    )
    sharingUser: User | None = Field(
        None, description="Output only. The user who shared the file with the requesting user, if applicable."
    )
    webViewLink: str | None = Field(
        None, description="Output only. A link for opening the file in a relevant Google editor or viewer in a browser."
    )
    webContentLink: str | None = Field(
        None,
        description="Output only. A link for downloading the content of the file in a browser. This is only available for files with binary content in Google Drive.",
    )
    size: int | None = Field(
        None,
        description="Output only. Size in bytes of blobs and Google Workspace editor files. Won't be populated for files that have no size, like shortcuts and folders.",
    )
    viewersCanCopyContent: bool | None = Field(
        None, description="Deprecated: Use `copyRequiresWriterPermission` instead."
    )
    permissions: list[Permission] | None = Field(
        None,
        description="Output only. The full list of permissions for the file. This is only available if the requesting user can share the file. Not populated for items in shared drives.",
    )
    hasThumbnail: bool | None = Field(
        None,
        description="Output only. Whether this file has a thumbnail. This doesn't indicate whether the requesting app has access to the thumbnail. To check access, look for the presence of the thumbnailLink field.",
    )
    spaces: list[str] | None = Field(
        None,
        description="Output only. The list of spaces which contain the file. The currently supported values are `drive`, `appDataFolder`, and `photos`.",
    )
    folderColorRgb: str | None = Field(
        None,
        description="The color for a folder or a shortcut to a folder as an RGB hex string. The supported colors are published in the `folderColorPalette` field of the [`about`](/workspace/drive/api/reference/rest/v3/about) resource. If an unsupported color is specified, the closest color in the palette is used instead.",
    )
    id: str | None = Field(None, description="The ID of the file.")
    name: str | None = Field(
        None,
        description="The name of the file. This isn't necessarily unique within a folder. Note that for immutable items such as the top-level folders of shared drives, the My Drive root folder, and the Application Data folder, the name is constant.",
    )
    description: str | None = Field(None, description="A short description of the file.")
    starred: bool | None = Field(None, description="Whether the user has starred the file.")
    trashed: bool | None = Field(
        None,
        description="Whether the file has been trashed, either explicitly or from a trashed parent folder. Only the owner may trash a file, and other users cannot see files in the owner's trash.",
    )
    explicitlyTrashed: bool | None = Field(
        None,
        description="Output only. Whether the file has been explicitly trashed, as opposed to recursively trashed from a parent folder.",
    )
    createdTime: AwareDatetime | None = Field(
        None, description="The time at which the file was created (RFC 3339 date-time)."
    )
    modifiedTime: AwareDatetime | None = Field(
        None,
        description="he last time the file was modified by anyone (RFC 3339 date-time). Note that setting modifiedTime will also update modifiedByMeTime for the user.",
    )
    modifiedByMeTime: AwareDatetime | None = Field(
        None, description="The last time the file was modified by the user (RFC 3339 date-time)."
    )
    viewedByMeTime: AwareDatetime | None = Field(
        None, description="The last time the file was viewed by the user (RFC 3339 date-time)."
    )
    sharedWithMeTime: AwareDatetime | None = Field(
        None, description="The time at which the file was shared with the user, if applicable (RFC 3339 date-time)."
    )
    quotaBytesUsed: int | None = Field(
        None,
        description="Output only. The number of storage quota bytes used by the file. This includes the head revision as well as previous revisions with `keepForever` enabled.",
    )
    version: int | None = Field(
        None,
        description="Output only. A monotonically increasing version number for the file. This reflects every change made to the file on the server, even those not visible to the user.",
    )
    originalFilename: str | None = Field(
        None,
        description="The original filename of the uploaded content if available, or else the original value of the `name` field. This is only available for files with binary content in Google Drive.",
    )
    ownedByMe: bool | None = Field(
        None, description="Output only. Whether the user owns the file. Not populated for items in shared drives."
    )
    fullFileExtension: str | None = Field(
        None,
        description="Output only. The full file extension extracted from the `name` field. May contain multiple concatenated extensions, such as \"tar.gz\". This is only available for files with binary content in Google Drive. This is automatically updated when the `name` field changes, however it's not cleared if the new name doesn't contain a valid extension.",
    )
    properties: dict[str, str] | None = Field(
        None,
        description="A collection of arbitrary key-value pairs which are visible to all apps.\nEntries with null values are cleared in update and copy requests.",
    )
    appProperties: dict[str, str] | None = Field(
        None,
        description="A collection of arbitrary key-value pairs which are private to the requesting app.\nEntries with null values are cleared in update and copy requests. These properties can only be retrieved using an authenticated request. An authenticated request uses an access token obtained with a OAuth 2 client ID. You cannot use an API key to retrieve private properties.",
    )
    isAppAuthorized: bool | None = Field(
        None, description="Output only. Whether the file was created or opened by the requesting app."
    )
    teamDriveId: str | None = Field(None, description="Deprecated: Output only. Use `driveId` instead.")
    capabilities: Capabilities | None = Field(
        None,
        description="Output only. Capabilities the current user has on this file. Each capability corresponds to a fine-grained action that a user may take. For more information, see [Understand file capabilities](https://developers.google.com/workspace/drive/api/guides/manage-sharing#capabilities).",
    )
    hasAugmentedPermissions: bool | None = Field(
        None,
        description="Output only. Whether there are permissions directly on this file. This field is only populated for items in shared drives.",
    )
    trashingUser: User | None = Field(
        None,
        description="Output only. If the file has been explicitly trashed, the user who trashed it. Only populated for items in shared drives.",
    )
    thumbnailVersion: int | None = Field(
        None, description="Output only. The thumbnail version for use in thumbnail cache invalidation."
    )
    trashedTime: AwareDatetime | None = Field(
        None,
        description="The time that the item was trashed (RFC 3339 date-time). Only populated for items in shared drives.",
    )
    modifiedByMe: bool | None = Field(None, description="Output only. Whether the file has been modified by this user.")
    permissionIds: list[str] | None = Field(
        None, description="Output only. List of permission IDs for users with access to this file."
    )
    imageMediaMetadata: ImageMediaMetadata | None = Field(
        None, description="Output only. Additional metadata about image media, if available."
    )
    videoMediaMetadata: VideoMediaMetadata | None = Field(
        None,
        description="Output only. Additional metadata about video media. This may not be available immediately upon upload.",
    )
    shortcutDetails: ShortcutDetails | None = Field(
        None,
        description="Shortcut file details. Only populated for shortcut files, which have the mimeType field set to `application/vnd.google-apps.shortcut`. Can only be set on `files.create` requests.",
    )
    contentRestrictions: list[ContentRestriction] | None = Field(
        None,
        description="Restrictions for accessing the content of the file. Only populated if such a restriction exists.",
    )
    resourceKey: str | None = Field(None, description="Output only. A key needed to access the item via a shared link.")
    linkShareMetadata: LinkShareMetadata | None = Field(
        None, description="Contains details about the link URLs that clients are using to refer to this item."
    )
    labelInfo: LabelInfo | None = Field(None, description="Output only. An overview of the labels on the file.")
    sha1Checksum: str | None = Field(
        None,
        description="Output only. The SHA1 checksum associated with this file, if available. This field is only populated for files with content stored in Google Drive; it's not populated for Docs Editors or shortcut files.",
    )
    sha256Checksum: str | None = Field(
        None,
        description="Output only. The SHA256 checksum associated with this file, if available. This field is only populated for files with content stored in Google Drive; it's not populated for Docs Editors or shortcut files.",
    )
    inheritedPermissionsDisabled: bool | None = Field(
        None,
        description="Whether this file has inherited permissions disabled. Inherited permissions are enabled by default.",
    )
    downloadRestrictions: DownloadRestrictionsMetadata | None = Field(
        None, description="Download restrictions applied on the file."
    )


class FileList(BaseModel):
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of files. This will be absent if the end of the files list has been reached. If the token is rejected for any reason, it should be discarded, and pagination should be restarted from the first page of results. The page token is typically valid for several hours. However, if new items are added or removed, your expected results might differ.",
    )
    kind: str | None = Field(
        "drive#fileList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#fileList"`.',
    )
    incompleteSearch: bool | None = Field(
        None,
        description="Whether the search process was incomplete. If true, then some search results might be missing, since all documents were not searched. This can occur when searching multiple drives with the `allDrives` corpora, but all corpora couldn't be searched. When this happens, it's suggested that clients narrow their query by choosing a different corpus such as `user` or `drive`.",
    )
    files: list[File] | None = Field(
        None,
        description="The list of files. If `nextPageToken` is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )


class ModifyLabelsRequest(BaseModel):
    labelModifications: list[LabelModification] | None = Field(
        None, description="The list of modifications to apply to the labels on the file."
    )
    kind: str | None = Field(None, description='This is always `"drive#modifyLabelsRequest"`.')


class Change(BaseModel):
    kind: str | None = Field(
        "drive#change",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#change"`.',
    )
    removed: bool | None = Field(
        None,
        description="Whether the file or shared drive has been removed from this list of changes, for example by deletion or loss of access.",
    )
    file: File | None = Field(
        None,
        description="The updated state of the file. Present if the type is file and the file has not been removed from this list of changes.",
    )
    fileId: str | None = Field(None, description="The ID of the file which has changed.")
    time: AwareDatetime | None = Field(None, description="The time of this change (RFC 3339 date-time).")
    driveId: str | None = Field(None, description="The ID of the shared drive associated with this change.")
    type: str | None = Field(None, description="Deprecated: Use `changeType` instead.")
    teamDriveId: str | None = Field(None, description="Deprecated: Use `driveId` instead.")
    teamDrive: TeamDrive | None = Field(None, description="Deprecated: Use `drive` instead.")
    changeType: str | None = Field(None, description="The type of the change. Possible values are `file` and `drive`.")
    drive: Drive | None = Field(
        None,
        description="The updated state of the shared drive. Present if the changeType is drive, the user is still a member of the shared drive, and the shared drive has not been deleted.",
    )


class ChangeList(BaseModel):
    kind: str | None = Field(
        "drive#changeList",
        description='Identifies what kind of resource this is. Value: the fixed string `"drive#changeList"`.',
    )
    nextPageToken: str | None = Field(
        None,
        description="The page token for the next page of changes. This will be absent if the end of the changes list has been reached. The page token doesn't expire.",
    )
    newStartPageToken: str | None = Field(
        None,
        description="The starting page token for future changes. This will be present only if the end of the current changes list has been reached. The page token doesn't expire.",
    )
    changes: list[Change] | None = Field(
        None,
        description="The list of changes. If nextPageToken is populated, then this list may be incomplete and an additional page of results should be fetched.",
    )
